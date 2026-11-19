/*
 * Copyright (c) 2026 Etnaviv Project
 * SPDX-License-Identifier: MIT
 */

/*
 * Run a compute workgroup with fewer hardware threads than invocations.
 *
 * On GC7000 r6204 a workgroup that uses barrier() must be resident in full:
 * invocations x temporaries <= 512 per shader core. NXP's compiler fits larger
 * groups by running several invocations in one hardware thread (with its
 * #sh_swWorkGroupIndex / #sh_hwWorkGroupIndex / #sh_threadMemAddr uniforms).
 * This pass does the same: with a fold factor F, a workgroup of N invocations
 * is dispatched as H = ceil(N / F) hardware threads, and every hardware thread
 * h runs the invocations h + k * H, k = 0..F-1, that are below N.
 *
 * In OpenGL ES 3.1, barrier() only appears in main(), outside control flow.
 * The shader is then a sequence of segments separated by barriers. Each
 * segment runs F times, once per invocation of the thread, with the
 * invocation IDs of that copy: unrolled, or in a loop over the copies when
 * the unrolled code would be large (the caller decides). The barrier between
 * two segments waits for all the hardware threads, so for all the invocations
 * of the group.
 *
 * Values that a segment takes from an earlier one are recomputed when that is
 * cheap (constants, uniforms, IDs and ALU or deref chains of those), or saved
 * by each invocation in a scratch buffer and loaded back: scratch base +
 * (slot * invocations of the dispatch + global invocation index) * 16.
 *
 * Each segment derives its IDs from its own opaque copies of the hardware
 * local ID and workgroup ID (nir_intrinsic_opaque_copy_etna): otherwise CSE
 * merges a recomputed value with the same one of an earlier segment, which
 * then stays live across the barrier and across all the copies.
 */

#include "etnaviv_nir.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "util/set.h"
#include "util/u_debug.h"

DEBUG_GET_ONCE_BOOL_OPTION(fold_unroll, "ETNA_FOLD_UNROLL", false)

#define SEG_NONE 0xff

struct fold_seg {
   nir_instr *barrier;   /* barrier after the segment, NULL for the last */
   nir_def *hwid, *wid;  /* opaque hardware local ID and workgroup ID */
   nir_def *lid, *lidx, *gid; /* placeholders for the invocation IDs */
   nir_instr *last_placeholder;
   struct hash_table *remap;
};

struct fold_state {
   nir_shader *s;
   nir_function_impl *impl;
   unsigned fold, size[3], n;
   unsigned num_slots;
   struct hash_table *slot_of;
   struct fold_seg *seg;
   unsigned num_segs;
};

static bool
is_cs_barrier(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_barrier &&
          nir_intrinsic_execution_scope(intr) >= SCOPE_WORKGROUP;
}

/* IDs, which each segment takes from its own values */
static bool
is_id(nir_def *def, nir_intrinsic_op *op)
{
   nir_instr *instr = nir_def_instr(def);

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   *op = nir_instr_as_intrinsic(instr)->intrinsic;
   return *op == nir_intrinsic_load_local_invocation_id ||
          *op == nir_intrinsic_load_local_invocation_index ||
          *op == nir_intrinsic_load_global_invocation_id ||
          *op == nir_intrinsic_load_workgroup_id;
}

/* can be cloned into a later segment: no memory access, same value in every
 * copy apart from the IDs, which the copies remap */
static bool
is_cloneable(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_alu:
   case nir_instr_type_deref:
      return true;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_num_workgroups:
      case nir_intrinsic_load_workgroup_size:
      case nir_intrinsic_load_uniform:
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_constant:
         return true;
      default:
         return false;
      }
   }
   default:
      return false;
   }
}

static nir_def *
global_index(nir_builder *b, struct fold_state *st, nir_def *lidx, nir_def *wid)
{
   nir_def *num = nir_load_num_workgroups(b);
   nir_def *group =
      nir_iadd(b, nir_channel(b, wid, 0),
               nir_imul(b, nir_channel(b, num, 0),
                        nir_iadd(b, nir_channel(b, wid, 1),
                                 nir_imul(b, nir_channel(b, num, 1), nir_channel(b, wid, 2)))));

   return nir_iadd(b, nir_imul_imm(b, group, st->n), lidx);
}

/* byte offset of a scratch slot of the invocation */
static nir_def *
scratch_offset(nir_builder *b, struct fold_state *st, struct fold_seg *seg, unsigned slot)
{
   nir_def *num = nir_load_num_workgroups(b);
   nir_def *total = nir_imul_imm(b, nir_imul(b, nir_imul(b, nir_channel(b, num, 0),
                                                         nir_channel(b, num, 1)),
                                             nir_channel(b, num, 2)), st->n);

   return nir_imul_imm(b, nir_iadd(b, nir_imul_imm(b, total, slot),
                                   global_index(b, st, seg->lidx, seg->wid)), 16);
}

/* slot of a value saved in scratch; stores it right after its definition */
static unsigned
save_value(struct fold_state *st, nir_def *def)
{
   struct hash_entry *e = _mesa_hash_table_search(st->slot_of, def);
   if (e)
      return (uintptr_t)e->data - 1;

   const unsigned slot = st->num_slots++;
   _mesa_hash_table_insert(st->slot_of, def, (void *)(uintptr_t)(slot + 1));

   nir_instr *instr = nir_def_instr(def);
   const unsigned seg = instr->pass_flags;
   nir_builder b = nir_builder_at(instr->type == nir_instr_type_phi ? nir_after_phis(instr->block)
                                                                   : nir_after_instr(instr));
   nir_def *value = def->bit_size == 1 ? nir_b2i32(&b, def) : nir_u2u32(&b, def);

   nir_store_global_etna(&b, value, nir_load_scratch_base_ptr(&b, 1, 32),
                         scratch_offset(&b, st, &st->seg[seg], slot),
                         .write_mask = BITFIELD_MASK(def->num_components));
   return slot;
}

static nir_def *value_in(struct fold_state *st, unsigned j, nir_def *def, unsigned depth,
                         nir_builder *b);

struct scratch_leaves_state {
   struct fold_state *st;
   unsigned j, depth;
   struct set *seen, *leaves;
};

static bool scratch_leaves_src(nir_src *src, void *data);

/* the values that value_in() would load from scratch to recompute def in
 * segment j (collected in l->leaves, stopping at two) */
static void
scratch_leaves(struct scratch_leaves_state *l, nir_def *def)
{
   struct fold_seg *seg = &l->st->seg[l->j];
   nir_instr *instr = nir_def_instr(def);
   nir_intrinsic_op op;

   if (l->leaves->entries > 1 || _mesa_set_search(l->seen, def))
      return;
   _mesa_set_add(l->seen, def);

   if (_mesa_hash_table_search(seg->remap, def) || is_id(def, &op) || instr->pass_flags == l->j ||
       (instr->type == nir_instr_type_intrinsic &&
        nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_opaque_copy_etna))
      return;

   if (is_cloneable(instr) && l->depth < 64) {
      l->depth++;
      nir_foreach_src(instr, scratch_leaves_src, l);
      l->depth--;
      return;
   }

   _mesa_set_add(l->leaves, def);
}

static bool
scratch_leaves_src(nir_src *src, void *data)
{
   scratch_leaves(data, src->ssa);
   return true;
}

/* An ALU value that would be recomputed from two or more values in scratch
 * gets a slot of its own: one store and one load instead of one of each per
 * source. Canny's stage1 (folded by 4) takes two values across its second
 * barrier that it computes from eight loads of the shared tile: 2 slots
 * instead of 8. */
static bool
save_whole(struct fold_state *st, unsigned j, nir_def *def, unsigned depth)
{
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   struct scratch_leaves_state l = {
      .st = st,
      .j = j,
      .depth = depth,
      .seen = _mesa_pointer_set_create(NULL),
      .leaves = _mesa_pointer_set_create(NULL),
   };
   scratch_leaves(&l, def);
   const bool whole = l.leaves->entries > 1;
   _mesa_set_destroy(l.seen, NULL);
   _mesa_set_destroy(l.leaves, NULL);
   return whole;
}

struct map_src_state {
   struct fold_state *st;
   unsigned j, depth;
   nir_builder *b;
};

static bool
map_src(nir_src *src, void *data)
{
   struct map_src_state *m = data;
   struct fold_seg *seg = &m->st->seg[m->j];

   if (!_mesa_hash_table_search(seg->remap, src->ssa))
      _mesa_hash_table_insert(seg->remap, src->ssa,
                              value_in(m->st, m->j, src->ssa, m->depth + 1, m->b));
   return true;
}

/* the value of def in segment j, built at b if it is new: at the top level of
 * the segment, right before its first use, so that it dominates the later
 * uses but does not stay live through the whole segment */
static nir_def *
value_in(struct fold_state *st, unsigned j, nir_def *def, unsigned depth, nir_builder *b)
{
   struct fold_seg *seg = &st->seg[j];
   struct hash_entry *e = _mesa_hash_table_search(seg->remap, def);
   nir_intrinsic_op op;

   if (e)
      return e->data;

   nir_instr *instr = nir_def_instr(def);
   nir_def *v;

   /* the opaque IDs of any segment: those of this one */
   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_opaque_copy_etna) {
      nir_def *src = nir_instr_as_intrinsic(instr)->src[0].ssa;
      return nir_def_instr(src)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == nir_intrinsic_load_workgroup_id
                ? seg->wid : seg->hwid;
   }

   if (is_id(def, &op)) {
      v = op == nir_intrinsic_load_local_invocation_id ? seg->lid :
          op == nir_intrinsic_load_local_invocation_index ? seg->lidx :
          op == nir_intrinsic_load_workgroup_id ? seg->wid : seg->gid;
      v = nir_u2uN(b, v, def->bit_size);
   } else if (instr->pass_flags == j) {
      return def;
   } else if (is_cloneable(instr) && depth < 64 && !save_whole(st, j, def, depth)) {
      struct map_src_state m = { st, j, depth, b };
      nir_foreach_src(instr, map_src, &m);
      nir_instr *clone = nir_instr_clone_deep(st->s, instr, seg->remap);
      nir_builder_instr_insert(b, clone);
      clone->pass_flags = j;
      v = nir_instr_def(clone);
   } else {
      const unsigned slot = save_value(st, def);
      v = nir_load_global_etna(b, def->num_components, 32, nir_load_scratch_base_ptr(b, 1, 32),
                               scratch_offset(b, st, seg, slot));
      if (def->bit_size == 1)
         v = nir_ine_imm(b, v, 0);
      else
         v = nir_u2uN(b, v, def->bit_size);
   }

   _mesa_hash_table_insert(seg->remap, def, v);
   return v;
}

/* top-level position before a use: before the instruction, or before the
 * top-level if or loop that contains it */
static nir_cursor
use_cursor(struct fold_state *st, nir_src *src)
{
   nir_cf_node *node;
   nir_instr *instr = NULL;

   if (nir_src_is_if(src)) {
      node = &nir_src_use_if(src)->cf_node;
   } else {
      instr = nir_src_use_instr(src);
      node = &instr->block->cf_node;
   }

   if (instr && node->parent == &st->impl->cf_node) {
      /* a phi of a top-level block follows the if or loop it merges */
      if (instr->type == nir_instr_type_phi)
         return nir_before_cf_node(nir_cf_node_prev(node));
      return nir_before_instr(instr);
   }

   while (node->parent != &st->impl->cf_node)
      node = node->parent;
   return nir_before_cf_node(node);
}

static bool
rewrite_src(nir_src *src, void *data)
{
   void **args = data;
   struct fold_state *st = args[0];
   const unsigned j = (uintptr_t)args[1];
   nir_intrinsic_op op;
   nir_instr *def_instr = nir_def_instr(src->ssa);

   if (def_instr->pass_flags != j || is_id(src->ssa, &op)) {
      nir_builder b = nir_builder_at(use_cursor(st, src));
      nir_def *v = value_in(st, j, src->ssa, 0, &b);
      if (v != src->ssa)
         nir_src_rewrite(src, v);
   }
   return true;
}

/* the instructions of the shader, in order, tagged with their segment;
 * returns false if a barrier is in control flow */
static bool
tag_segments(struct fold_state *st)
{
   unsigned seg = 0, barriers = 0;

   nir_foreach_block(block, st->impl) {
      const bool top = block->cf_node.parent == &st->impl->cf_node;

      nir_foreach_instr(instr, block) {
         instr->pass_flags = seg;
         if (is_cs_barrier(instr)) {
            if (!top)
               return false;
            instr->pass_flags = SEG_NONE;
            seg++;
            barriers++;
         }
      }
   }

   st->num_segs = barriers + 1;
   return true;
}

/* tag every instruction with the segment of its position: value_in() and
 * save_value() add instructions that are not tagged yet (0), and a later
 * segment must not take one of them for a value of segment 0 */
static void
retag_segments(struct fold_state *st)
{
   unsigned seg = 0;

   nir_foreach_block(block, st->impl) {
      nir_foreach_instr(instr, block) {
         if (instr->pass_flags != SEG_NONE)
            instr->pass_flags = seg;
         else if (is_cs_barrier(instr))
            seg++;
      }
   }
}

bool
etna_nir_fold_workgroup(nir_shader *s, unsigned fold, bool loop_copies, unsigned *scratch_slots)
{
   struct fold_state st = {
      .s = s,
      .impl = nir_shader_get_entrypoint(s),
      .fold = fold,
      .size = { s->info.workgroup_size[0], s->info.workgroup_size[1], s->info.workgroup_size[2] },
   };

   st.n = st.size[0] * st.size[1] * st.size[2];
   *scratch_slots = 0;

   if (s->info.workgroup_size_variable || fold < 2 || fold > st.n)
      return false;

   NIR_PASS(_, s, nir_lower_returns);
   st.impl = nir_shader_get_entrypoint(s);

   void *mem_ctx = ralloc_context(NULL);

   if (!tag_segments(&st)) {
      ralloc_free(mem_ctx);
      return false;
   }

   st.slot_of = _mesa_pointer_hash_table_create(mem_ctx);
   st.seg = rzalloc_array(mem_ctx, struct fold_seg, st.num_segs);

   /* barriers in order */
   unsigned n = 0;
   nir_foreach_block(block, st.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->pass_flags == SEG_NONE)
            st.seg[n++].barrier = instr;
      }
   }

   /* placeholders for the IDs at the start of each segment */
   for (unsigned j = 0; j < st.num_segs; j++) {
      struct fold_seg *seg = &st.seg[j];
      nir_builder b = nir_builder_at(j ? nir_after_instr(st.seg[j - 1].barrier)
                                       : nir_before_impl(st.impl));

      nir_def *hwid = nir_load_local_invocation_id(&b);
      nir_def *wid = nir_load_workgroup_id(&b);
      seg->hwid = nir_opaque_copy_etna(&b, hwid);
      seg->wid = nir_opaque_copy_etna(&b, wid);
      nir_def_instr(hwid)->pass_flags = SEG_NONE;
      nir_def_instr(wid)->pass_flags = SEG_NONE;
      nir_def_instr(seg->hwid)->pass_flags = SEG_NONE;
      nir_def_instr(seg->wid)->pass_flags = SEG_NONE;
      seg->lid = nir_load_local_invocation_id(&b);
      seg->lidx = nir_load_local_invocation_index(&b);
      seg->gid = nir_load_global_invocation_id(&b, 32);
      nir_def_instr(seg->lid)->pass_flags = SEG_NONE;
      nir_def_instr(seg->lidx)->pass_flags = SEG_NONE;
      nir_def_instr(seg->gid)->pass_flags = SEG_NONE;
      seg->last_placeholder = nir_def_instr(seg->gid);
      seg->remap = _mesa_pointer_hash_table_create(mem_ctx);
   }

   /* every segment uses only its own values and its ID placeholders; walk
    * it in program order, if conditions at the end of the block before */
   for (unsigned j = 0; j < st.num_segs; j++) {
      void *args[2] = { &st, (void *)(uintptr_t)j };
      nir_instr *end = st.seg[j].barrier;
      nir_block *blk = st.seg[j].last_placeholder->block;
      nir_instr *instr = nir_instr_next(st.seg[j].last_placeholder);

      for (;;) {
         for (; instr && instr != end; ) {
            nir_instr *next = nir_instr_next(instr);
            if (instr->pass_flags == j)
               nir_foreach_src(instr, rewrite_src, args);
            instr = next;
         }
         if (instr == end && end)
            break;

         nir_if *nif = nir_block_get_following_if(blk);
         if (nif)
            rewrite_src(&nif->condition, args);

         blk = nir_block_cf_tree_next(blk);
         if (!blk)
            break;
         instr = nir_block_first_instr(blk);
      }

      retag_segments(&st);
   }

   /* run every segment once per invocation of the thread: unrolled, or in a
    * loop over the copies (ETNA_FOLD_UNROLL=1 always unrolls); when F
    * doesn't divide N, the copies past the end of the workgroup are skipped */
   const unsigned hw = DIV_ROUND_UP(st.n, fold);
   const bool loop_fold = loop_copies && !debug_get_option_fold_unroll();

   for (unsigned j = 0; j < st.num_segs; j++) {
      struct fold_seg *seg = &st.seg[j];
      nir_cf_list list;

      nir_cf_extract(&list, nir_after_instr(seg->last_placeholder),
                     seg->barrier ? nir_before_instr(seg->barrier) : nir_after_impl(st.impl));

      if (loop_fold) {
         nir_builder b = nir_builder_at(nir_after_instr(seg->last_placeholder));
         nir_variable *kv = nir_local_variable_create(st.impl, glsl_uint_type(), "fold_copy");

         nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
         nir_loop *loop = nir_push_loop(&b);
         /* nir_opt_loop_unroll would make the unrolled copies again */
         loop->control = nir_loop_control_dont_unroll;
         {
            nir_def *k = nir_load_var(&b, kv);
            nir_break_if(&b, nir_uge_imm(&b, k, fold));
            nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);

            nir_def *l = nir_iadd(&b, nir_channel(&b, seg->hwid, 0), nir_imul_imm(&b, k, hw));
            nir_def *lid = nir_vec3(&b, nir_umod_imm(&b, l, st.size[0]),
                                    nir_umod_imm(&b, nir_udiv_imm(&b, l, st.size[0]), st.size[1]),
                                    nir_udiv_imm(&b, l, st.size[0] * st.size[1]));
            nir_def *gid = nir_iadd(&b, nir_imul(&b, seg->wid,
                                                 nir_imm_ivec3(&b, st.size[0], st.size[1], st.size[2])),
                                    lid);
            struct hash_table *remap = _mesa_pointer_hash_table_create(mem_ctx);

            _mesa_hash_table_insert(remap, seg->lid, nir_u2uN(&b, lid, seg->lid->bit_size));
            _mesa_hash_table_insert(remap, seg->lidx, nir_u2uN(&b, l, seg->lidx->bit_size));
            _mesa_hash_table_insert(remap, seg->gid, gid);

            if (hw * fold > st.n) {
               nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, l, st.n));
               nir_cf_list_clone_and_reinsert(&list, &nif->cf_node, b.cursor, remap);
               nir_pop_if(&b, nif);
            } else {
               nir_cf_list_clone_and_reinsert(&list, &loop->cf_node, b.cursor, remap);
            }
         }
         nir_pop_loop(&b, loop);
         nir_cf_delete(&list);
         continue;
      }

      for (int k = fold - 1; k >= 0; k--) {
         nir_builder b = nir_builder_at(nir_after_instr(seg->last_placeholder));
         nir_def *l = nir_iadd_imm(&b, nir_channel(&b, seg->hwid, 0), k * hw);
         nir_def *lid = nir_vec3(&b, nir_umod_imm(&b, l, st.size[0]),
                                 nir_umod_imm(&b, nir_udiv_imm(&b, l, st.size[0]), st.size[1]),
                                 nir_udiv_imm(&b, l, st.size[0] * st.size[1]));
         nir_def *gid = nir_iadd(&b, nir_imul(&b, seg->wid,
                                              nir_imm_ivec3(&b, st.size[0], st.size[1], st.size[2])),
                                 lid);
         struct hash_table *remap = _mesa_pointer_hash_table_create(mem_ctx);

         _mesa_hash_table_insert(remap, seg->lid, nir_u2uN(&b, lid, seg->lid->bit_size));
         _mesa_hash_table_insert(remap, seg->lidx, nir_u2uN(&b, l, seg->lidx->bit_size));
         _mesa_hash_table_insert(remap, seg->gid, gid);

         if ((k + 1) * hw > st.n) {
            nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, l, st.n));
            nir_cf_list_clone_and_reinsert(&list, &nif->cf_node, b.cursor, remap);
            nir_pop_if(&b, nif);
         } else {
            nir_cf_list_clone_and_reinsert(&list, &st.impl->cf_node, b.cursor, remap);
         }
      }

      nir_cf_delete(&list);
   }

   for (unsigned j = 0; j < st.num_segs; j++) {
      nir_instr_remove(nir_def_instr(st.seg[j].lid));
      nir_instr_remove(nir_def_instr(st.seg[j].lidx));
      nir_instr_remove(nir_def_instr(st.seg[j].gid));
   }

   *scratch_slots = st.num_slots;
   ralloc_free(mem_ctx);

   nir_progress(true, st.impl, nir_metadata_none);
   /* the loop counters */
   if (loop_fold)
      NIR_PASS(_, s, nir_lower_vars_to_ssa);
   return true;
}
