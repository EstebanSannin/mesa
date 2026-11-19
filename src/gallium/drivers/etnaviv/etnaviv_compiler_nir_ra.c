/*
 * Copyright (c) 2019 Zodiac Inflight Innovations
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *    Jonathan Marek <jonathan@marek.ca>
 */

#include "etnaviv_compiler_nir.h"
#include "util/register_allocate.h"

/* use "r63.z" for depth reg, it will wrap around to r0.z by reg_get_base
 * (fs registers are offset by 1 to avoid reserving r0)
 */
#define REG_FRAG_DEPTH ((ETNA_MAX_TEMPS - 1) * NUM_REG_TYPES + REG_TYPE_VIRT_SCALAR_Z)

/* precomputed by register_allocate  */
static unsigned int *q_values[] = {
   (unsigned int[]) {1, 2, 3, 4, 2, 2, 3, },
   (unsigned int[]) {3, 5, 6, 6, 5, 5, 6, },
   (unsigned int[]) {3, 4, 4, 4, 4, 4, 4, },
   (unsigned int[]) {1, 1, 1, 1, 1, 1, 1, },
   (unsigned int[]) {1, 2, 2, 2, 1, 2, 2, },
   (unsigned int[]) {2, 3, 3, 3, 2, 3, 3, },
   (unsigned int[]) {2, 2, 2, 2, 2, 2, 2, },
};

static inline int reg_get_class(int virt_reg)
{
   switch (reg_get_type(virt_reg)) {
   case REG_TYPE_VEC4:
      return REG_CLASS_VEC4;
   case REG_TYPE_VIRT_VEC3_XYZ:
   case REG_TYPE_VIRT_VEC3_XYW:
   case REG_TYPE_VIRT_VEC3_XZW:
   case REG_TYPE_VIRT_VEC3_YZW:
      return REG_CLASS_VIRT_VEC3;
   case REG_TYPE_VIRT_VEC2_XY:
   case REG_TYPE_VIRT_VEC2_XZ:
   case REG_TYPE_VIRT_VEC2_XW:
   case REG_TYPE_VIRT_VEC2_YZ:
   case REG_TYPE_VIRT_VEC2_YW:
   case REG_TYPE_VIRT_VEC2_ZW:
      return REG_CLASS_VIRT_VEC2;
   case REG_TYPE_VIRT_SCALAR_X:
   case REG_TYPE_VIRT_SCALAR_Y:
   case REG_TYPE_VIRT_SCALAR_Z:
   case REG_TYPE_VIRT_SCALAR_W:
      return REG_CLASS_VIRT_SCALAR;
   case REG_TYPE_VIRT_VEC2T_XY:
   case REG_TYPE_VIRT_VEC2T_ZW:
      return REG_CLASS_VIRT_VEC2T;
   case REG_TYPE_VIRT_VEC2C_XY:
   case REG_TYPE_VIRT_VEC2C_YZ:
   case REG_TYPE_VIRT_VEC2C_ZW:
      return REG_CLASS_VIRT_VEC2C;
   case REG_TYPE_VIRT_VEC3C_XYZ:
   case REG_TYPE_VIRT_VEC3C_YZW:
      return REG_CLASS_VIRT_VEC3C;
   }

   assert(false);
   return 0;
}

struct ra_regs *
etna_ra_setup(void *mem_ctx)
{
   struct ra_regs *regs = ra_alloc_reg_set(mem_ctx, ETNA_MAX_TEMPS *
                  NUM_REG_TYPES, false);

   /* classes always be created from index 0, so equal to the class enum
    * which represents a register with (c+1) components
    */
   struct ra_class *classes[NUM_REG_CLASSES];
   for (int c = 0; c < NUM_REG_CLASSES; c++)
      classes[c] = ra_alloc_reg_class(regs);
   /* add each register of each class */
   for (int r = 0; r < NUM_REG_TYPES * ETNA_MAX_TEMPS; r++)
      ra_class_add_reg(classes[reg_get_class(r)], r);
   /* set conflicts */
   for (int r = 0; r < ETNA_MAX_TEMPS; r++) {
      for (int i = 0; i < NUM_REG_TYPES; i++) {
         for (int j = 0; j < i; j++) {
            if (reg_writemask[i] & reg_writemask[j]) {
               ra_add_reg_conflict(regs, NUM_REG_TYPES * r + i,
                                         NUM_REG_TYPES * r + j);
            }
         }
      }
   }
   ra_set_finalize(regs, q_values);

   return regs;
}

/* the values the components of an image coordinate come from: the sources
 * of its vecN or mov, through the movs that lower_alu() put in front of a
 * vecN; else the coordinate itself */
static unsigned
image_coord_roots(nir_def *coord, nir_def **roots)
{
   unsigned n = 0;

   if (nir_def_is_alu(coord)) {
      nir_alu_instr *alu = nir_def_as_alu(coord);

      if (nir_op_is_vec(alu->op) || alu->op == nir_op_mov) {
         for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs && n < 4; s++) {
            nir_def *d = alu->src[s].src.ssa;

            if (nir_def_is_alu(d) && nir_def_as_alu(d)->op == nir_op_mov)
               d = nir_def_as_alu(d)->src[0].src.ssa;
            roots[n++] = d;
         }
         return n;
      }
   }
   roots[n++] = coord;
   return n;
}

/* img_loads whose coordinate shares a component value with the coordinate
 * of another image access: their results get no register of the coordinate
 * (see etna_ra_assign()) */
static struct set *
shared_coord_loads(nir_function_impl *impl)
{
   struct hash_table *count = _mesa_pointer_hash_table_create(NULL);
   struct set *loads = _mesa_pointer_set_create(NULL);

   for (unsigned pass = 0; pass < 2; pass++) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_image_etna &&
                intr->intrinsic != nir_intrinsic_store_image_etna)
               continue;

            nir_def *roots[4];
            const unsigned n = image_coord_roots(intr->src[1].ssa, roots);

            for (unsigned i = 0; i < n; i++) {
               bool dup = false;
               for (unsigned j = 0; j < i; j++)
                  dup |= roots[j] == roots[i];
               if (dup)
                  continue;

               struct hash_entry *e = _mesa_hash_table_search(count, roots[i]);
               if (pass == 0) {
                  if (e)
                     e->data = (void *)((uintptr_t)e->data + 1);
                  else
                     _mesa_hash_table_insert(count, roots[i], (void *)(uintptr_t)1);
               } else if (intr->intrinsic == nir_intrinsic_load_image_etna && e &&
                          (uintptr_t)e->data > 1) {
                  _mesa_set_add(loads, intr);
               }
            }
         }
      }
   }

   _mesa_hash_table_destroy(count, NULL);
   return loads;
}

bool
etna_ra_assign(struct etna_compile *c, nir_shader *shader)
{
   struct etna_compiler *compiler = c->variant->shader->compiler;
   struct ra_regs *regs = compiler->regs;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   /* liveness and interference */

   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            instr->pass_flags &= ~PASS_FLAGS_IS_DEAD_MASK;
         }
      }
   }

   /* this gives an approximation/upper limit on how many nodes are needed
    * (some ssa values do not represent an allocated register)
    */
   unsigned max_nodes = impl->ssa_alloc;
   unsigned *live_map = ralloc_array(NULL, unsigned, max_nodes);
   memset(live_map, 0xff, sizeof(unsigned) * max_nodes);
   struct live_def *defs = rzalloc_array(NULL, struct live_def, max_nodes);

   unsigned num_nodes = etna_live_defs(impl, defs, live_map);
   struct ra_graph *g = ra_alloc_interference_graph(regs, num_nodes);

   /* set classes from num_components */
   for (unsigned i = 0; i < num_nodes; i++) {
      nir_instr *instr = defs[i].instr;
      nir_def *def = defs[i].def;
      unsigned comp = def->num_components - 1;

      if (instr->type == nir_instr_type_alu &&
         etna_core_has_feature(c->info, ETNA_FEATURE_HAS_FAST_TRANSCENDENTALS)) {
         switch (nir_instr_as_alu(instr)->op) {
         case nir_op_fdiv:
         case nir_op_flog2:
         case nir_op_fsin:
         case nir_op_fcos:
            comp = REG_CLASS_VIRT_VEC2T;
            break;
         default:
            break;
         }
      }

      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         /* can't have dst swizzle or sparse writemask on UBO and constant loads */
         if (intr->intrinsic == nir_intrinsic_load_ubo ||
             intr->intrinsic == nir_intrinsic_load_global_etna ||
             intr->intrinsic == nir_intrinsic_load_constant) {
            assert(def == &intr->def);
            if (def->num_components == 2)
               comp = REG_CLASS_VIRT_VEC2C;
            if (def->num_components == 3)
               comp = REG_CLASS_VIRT_VEC3C;
         }
      }

      ra_set_node_class(g, i, ra_get_class_from_index(regs, comp));
   }

   struct set *shared_loads = shader->info.stage == MESA_SHADER_COMPUTE ?
                              shared_coord_loads(impl) : NULL;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_def *def = def_for_instr(instr);
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         unsigned reg;

         switch (intr->intrinsic) {
         case nir_intrinsic_store_deref: {
            /* don't want outputs to be swizzled
             * TODO: better would be to set the type to X/XY/XYZ/XYZW
             * TODO: what if fragcoord.z is read after writing fragdepth?
             */
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            unsigned index = live_map[src_index(impl, &intr->src[1])];

            if (shader->info.stage == MESA_SHADER_FRAGMENT &&
                deref->var->data.location == FRAG_RESULT_DEPTH) {
               ra_set_node_reg(g, index, REG_FRAG_DEPTH);
            } else {
               ra_set_node_class(g, index, ra_get_class_from_index(regs, REG_CLASS_VEC4));
            }
         } continue;
         case nir_intrinsic_load_image_etna: {
            /* compute: the result of a load whose coordinate shares a
             * component with another image access never takes the register
             * of its coordinate, which the next access can then reuse with
             * the components that don't change (drop_redundant_movs()) */
            unsigned index = live_map[def_index(impl, def)];
            if (shared_loads && _mesa_set_search(shared_loads, intr) && defs[index].live_start)
               defs[index].live_start--;
         } continue;
         case nir_intrinsic_global_atomic_swap_etna: {
            /* atomic_cmp_xchg reads the new and the compared value from .x
             * and .y of its destination register, which is written before
             * the instruction (see emit): give the result a full register,
             * live from just before the instruction so that it shares no
             * register with the sources. */
            unsigned index = live_map[def_index(impl, def)];
            ra_set_node_class(g, index, ra_get_class_from_index(regs, REG_CLASS_VEC4));
            if (defs[index].live_start)
               defs[index].live_start--;
         } continue;
         case nir_intrinsic_load_input:
            reg = nir_intrinsic_base(intr) * NUM_REG_TYPES + (unsigned[]) {
               REG_TYPE_VIRT_SCALAR_X,
               REG_TYPE_VIRT_VEC2_XY,
               REG_TYPE_VIRT_VEC3_XYZ,
               REG_TYPE_VEC4,
            }[def->num_components - 1];
            break;
         case nir_intrinsic_load_instance_id:
            reg = c->variant->infile.num_reg * NUM_REG_TYPES + REG_TYPE_VIRT_SCALAR_Y;
            break;
         case nir_intrinsic_load_vertex_id:
            reg = c->variant->infile.num_reg * NUM_REG_TYPES + REG_TYPE_VIRT_SCALAR_X;
            break;
         default:
            continue;
         }

         ra_set_node_reg(g, live_map[def_index(impl, def)], reg);
      }
   }

   if (shared_loads)
      _mesa_set_destroy(shared_loads, NULL);

   /* a def nobody reads (an atomic's old value) is still written: keep its
    * register from overlapping values that are live across that instruction */
   for (unsigned i = 0; i < num_nodes; i++) {
      if (defs[i].live_end <= defs[i].live_start)
         defs[i].live_end = defs[i].live_start + 1;
   }

   /* add interference for intersecting live ranges */
   for (unsigned i = 0; i < num_nodes; i++) {
      assert(defs[i].live_start < defs[i].live_end);
      for (unsigned j = 0; j < i; j++) {
         if (defs[i].live_start >= defs[j].live_end || defs[j].live_start >= defs[i].live_end)
            continue;
         ra_add_node_interference(g, i, j);
      }
   }

   ralloc_free(defs);

   /* Allocate registers. There is no spilling: a shader that doesn't fit
    * fails to compile. */
   bool ok = ra_allocate(g);

   c->g = g;
   c->live_map = live_map;
   c->num_nodes = num_nodes;

   return ok;
}

unsigned
etna_ra_finish(struct etna_compile *c)
{
   /* TODO: better way to get number of registers used? */
   unsigned j = 0;
   for (unsigned i = 0; i < c->num_nodes; i++) {
      j = MAX2(j, reg_get_base(c, ra_get_node_reg(c->g, i)) + 1);
   }

   ralloc_free(c->g);
   ralloc_free(c->live_map);

   return j;
}
