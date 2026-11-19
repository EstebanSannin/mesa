/*
 * Copyright (c) 2020 Etnaviv Project
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

#ifndef H_ETNAVIV_COMPILER_NIR
#define H_ETNAVIV_COMPILER_NIR

#include "compiler/nir/nir.h"
#include "etna_core_info.h"
#include "etnaviv_asm.h"
#include "etnaviv_compiler.h"
#include "util/compiler.h"
#include "util/log.h"
#include "util/macros.h"

/* dual16: what a register component holds (dual16_rewrite()) */
#define ETNA_D16_P 1 /* a half per work-item, packed: work-item 0 in bits 0-15 */
#define ETNA_D16_W 2 /* a 32-bit value per work-item */

struct etna_d16_inst {
   uint8_t dst;                /* kind of the result */
   uint8_t src[ETNA_NUM_SRC];  /* kinds of the register sources (0: not known) */
   bool from_nir;              /* emitted for an ALU instruction or a store */
};

struct etna_compile {
   nir_shader *nir;
   nir_function_impl *impl;
#define is_fs(c) ((c)->nir->info.stage == MESA_SHADER_FRAGMENT)
   const struct etna_core_info *info;
   const struct etna_specs *specs;
   struct etna_shader_variant *variant;

   /* block # to instr index */
   unsigned *block_ptr;

   /* Code generation: HALTI5 fetches shaders from memory, so the code grows
    * as needed, up to ETNA_MAX_SHADER_INSTRUCTIONS */
   int inst_ptr; /* current instruction pointer */
   struct etna_inst *code;
   unsigned code_alloc;

   /* constants */
   uint64_t consts[ETNA_MAX_IMM];
   unsigned const_count;

   /* movs of constant sources inserted by lower_alu in the current block */
   struct {
      nir_def *src, *mov;
   } const_movs[16];
   unsigned num_const_movs;

   /* ra state */
   struct ra_graph *g;
   unsigned *live_map;
   unsigned num_nodes;

   /* There was an error during compilation */
   bool error;

   /* dual16: per emitted instruction, what its result and register sources
    * hold, from the registers of the NIR sources of the instruction being
    * emitted (dual16_record()) */
   struct etna_d16_inst *d16;
   uint8_t d16_dst;
   bool d16_from_nir;
   unsigned d16_nrec;
   struct {
      uint8_t reg, mask, kind;
   } d16_rec[8];
};

#define compile_error(ctx, args...) ({ \
   mesa_loge(args); \
   ctx->error = true; \
   abort(); \
})

enum etna_pass_flags {
   BYPASS_DST = BITFIELD_BIT(0),
   BYPASS_SRC = BITFIELD_BIT(1),

   /* source modifier */
   SRC0_MOD_NEG = BITFIELD_BIT(2),
   SRC1_MOD_NEG = BITFIELD_BIT(3),
   SRC2_MOD_NEG = BITFIELD_BIT(4),
   SRC0_MOD_ABS = BITFIELD_BIT(5),
   SRC1_MOD_ABS = BITFIELD_BIT(6),
   SRC2_MOD_ABS = BITFIELD_BIT(7),
};

#define PASS_FLAGS_IS_DEAD_MASK     BITFIELD_RANGE(0, 2)
#define PASS_FLAGS_SRC_MOD_NEG_MASK BITFIELD_RANGE(2, 3)
#define PASS_FLAGS_SRC_MOD_ABS_MASK BITFIELD_RANGE(5, 3)

static_assert(PASS_FLAGS_IS_DEAD_MASK == (BYPASS_DST | BYPASS_SRC), "is_dead_mask is wrong");
static_assert(PASS_FLAGS_SRC_MOD_NEG_MASK == (SRC0_MOD_NEG | SRC1_MOD_NEG | SRC2_MOD_NEG), "src_mod_neg_mask is wrong");
static_assert(PASS_FLAGS_SRC_MOD_ABS_MASK == (SRC0_MOD_ABS | SRC1_MOD_ABS | SRC2_MOD_ABS), "src_mod_abs_mask is wrong");

static inline bool is_dead_instruction(nir_instr *instr)
{
   return instr->pass_flags & PASS_FLAGS_IS_DEAD_MASK;
}

static inline void set_src_mod_abs(nir_instr *instr, unsigned idx)
{
   assert(idx < 3);
   instr->pass_flags |= (SRC0_MOD_ABS << idx);
}

static inline void set_src_mod_neg(nir_instr *instr, unsigned idx)
{
   assert(idx < 3);
   instr->pass_flags |= (SRC0_MOD_NEG << idx);
}

static inline void toggle_src_mod_neg(nir_instr *instr, unsigned idx)
{
   assert(idx < 3);
   instr->pass_flags ^= (SRC0_MOD_NEG << idx);
}

static inline bool is_src_mod_abs(nir_instr *instr, unsigned idx)
{
   if (idx < 3)
      return instr->pass_flags & (SRC0_MOD_ABS << idx);

   return false;
}

static inline bool is_src_mod_neg(nir_instr *instr, unsigned idx)
{
   if (idx < 3)
      return instr->pass_flags & (SRC0_MOD_NEG << idx);

   return false;
}

static inline bool is_sysval(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   return intr->intrinsic == nir_intrinsic_load_front_face ||
          intr->intrinsic == nir_intrinsic_load_frag_coord;
}

/* get unique ssa/reg index for nir_src */
static inline unsigned
src_index(nir_function_impl *impl, nir_src *src)
{
   nir_intrinsic_instr *load = nir_load_reg_for_def(src->ssa);

   if (load) {
      nir_def *reg = load->src[0].ssa;
      ASSERTED nir_intrinsic_instr *decl = nir_reg_get_decl(reg);
      assert(nir_intrinsic_base(load) == 0);
      assert(nir_intrinsic_num_array_elems(decl) == 0);

      return reg->index;
   }

   return src->ssa->index;
}

/* get unique ssa/reg index for nir_def */
static inline unsigned
def_index(nir_function_impl *impl, nir_def *def)
{
   nir_intrinsic_instr *store = nir_store_reg_for_def(def);

   if (store) {
      nir_def *reg = store->src[1].ssa;
      ASSERTED nir_intrinsic_instr *decl = nir_reg_get_decl(reg);
      assert(nir_intrinsic_base(store) == 0);
      assert(nir_intrinsic_num_array_elems(decl) == 0);

      return reg->index;
   }

   return def->index;
}

static inline void
update_swiz_mask(nir_alu_instr *alu, nir_def *def, unsigned *swiz, unsigned *mask)
{
   if (!swiz)
      return;

   bool is_vec = def != NULL;
   unsigned swizzle = 0, write_mask = 0;
   for (unsigned i = 0; i < alu->def.num_components; i++) {
      /* src is different (only check for vecN) */
      if (is_vec && alu->src[i].src.ssa != def)
         continue;

      unsigned src_swiz = is_vec ? alu->src[i].swizzle[0] : alu->src[0].swizzle[i];
      swizzle |= (*swiz >> src_swiz * 2 & 3) << i * 2;
      /* this channel isn't written through this chain */
      if (*mask & (1 << src_swiz))
         write_mask |= 1 << i;
   }
   *swiz = swizzle;
   *mask = write_mask;
}

static nir_def *
real_def(nir_def *def, unsigned *swiz, unsigned *mask)
{
   if (!def)
      return def;

   bool can_bypass_src = !nir_def_used_by_if(def);
   nir_instr *p_instr = nir_def_instr(def);

   /* an intrinsic result nobody reads (an atomic's old value) is still
    * written by the instruction, so it needs its own register */
   if (nir_def_is_unused(def) && p_instr->type == nir_instr_type_intrinsic)
      return def;

   /* if used by a vecN, the "real" destination becomes the vecN destination
    * lower_alu guarantees that values used by a vecN are only used by that vecN
    * we can apply the same logic to movs in a some cases too
    */
   nir_foreach_use(use_src, def) {
      nir_instr *instr = nir_src_use_instr(use_src);

      /* src bypass check: for now only deal with tex src mov case
       * note: for alu don't bypass mov for multiple uniform sources
       */
      switch (instr->type) {
      case nir_instr_type_intrinsic: {
         /* the coordinate of img_load and img_store takes a swizzle too, of
          * a register: a mov of an ALU result needs no instruction */
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

         if ((intr->intrinsic == nir_intrinsic_load_image_etna ||
              intr->intrinsic == nir_intrinsic_store_image_etna) &&
             use_src == &intr->src[1] && p_instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(p_instr)->op == nir_op_mov &&
             nir_def_instr_type(nir_instr_as_alu(p_instr)->src[0].src.ssa) == nir_instr_type_alu)
            break;

         can_bypass_src = false;
         break;
      }
      case nir_instr_type_tex:
         if (p_instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(p_instr)->op == nir_op_mov) {
            break;
         }
         FALLTHROUGH;
      default:
         can_bypass_src = false;
         break;
      }

      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
         assert(!nir_def_used_by_if(def));
         nir_foreach_use(use_src, def)
            assert(nir_src_use_instr(use_src) == instr);

         update_swiz_mask(alu, def, swiz, mask);
         break;
      case nir_op_mov: {
         switch (nir_def_instr_type(def)) {
         case nir_instr_type_alu:
         case nir_instr_type_tex:
            break;
         default:
            continue;
         }
         if (nir_def_used_by_if(def) || list_length(&def->uses) > 1)
            continue;

         update_swiz_mask(alu, NULL, swiz, mask);
         break;
      };
      default:
         continue;
      }

      assert(!(instr->pass_flags & BYPASS_SRC));
      instr->pass_flags |= BYPASS_DST;
      return real_def(&alu->def, swiz, mask);
   }

   if (can_bypass_src && !(p_instr->pass_flags & BYPASS_DST)) {
      p_instr->pass_flags |= BYPASS_SRC;
      return NULL;
   }

   return def;
}

/* if instruction dest needs a register, return nir_def for it */
static inline nir_def *
def_for_instr(nir_instr *instr)
{
   nir_def *def = NULL;

   switch (instr->type) {
   case nir_instr_type_alu:
      def = &nir_instr_as_alu(instr)->def;
      break;
   case nir_instr_type_tex:
      def = &nir_instr_as_tex(instr)->def;
      break;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_uniform ||
          intr->intrinsic == nir_intrinsic_load_ubo ||
          intr->intrinsic == nir_intrinsic_load_global_etna ||
          intr->intrinsic == nir_intrinsic_load_image_etna ||
          intr->intrinsic == nir_intrinsic_global_atomic_etna ||
          intr->intrinsic == nir_intrinsic_global_atomic_swap_etna ||
          intr->intrinsic == nir_intrinsic_load_constant ||
          intr->intrinsic == nir_intrinsic_load_input ||
          intr->intrinsic == nir_intrinsic_load_instance_id ||
          intr->intrinsic == nir_intrinsic_load_vertex_id ||
          intr->intrinsic == nir_intrinsic_ddx ||
          intr->intrinsic == nir_intrinsic_ddy)
         def = &intr->def;
   } break;
   case nir_instr_type_deref:
      return NULL;
   default:
      break;
   }
   return real_def(def, NULL, NULL);
}

struct live_def {
   nir_instr *instr;
   nir_def *def; /* cached def_for_instr */
   unsigned live_start, live_end; /* live range */
};

unsigned
etna_live_defs(nir_function_impl *impl, struct live_def *defs, unsigned *live_map);

/* Swizzles and write masks can be used to layer virtual non-interfering
 * registers on top of the real VEC4 registers. For example, the virtual
 * VEC3_XYZ register and the virtual SCALAR_W register that use the same
 * physical VEC4 base register do not interfere.
 */
enum reg_class {
   REG_CLASS_VIRT_SCALAR,
   REG_CLASS_VIRT_VEC2,
   REG_CLASS_VIRT_VEC3,
   REG_CLASS_VEC4,
   /* special vec2 class for fast transcendentals, limited to XY or ZW */
   REG_CLASS_VIRT_VEC2T,
   /* special classes for LOAD - contiguous components */
   REG_CLASS_VIRT_VEC2C,
   REG_CLASS_VIRT_VEC3C,
   NUM_REG_CLASSES,
};

enum reg_type {
   REG_TYPE_VEC4,
   REG_TYPE_VIRT_VEC3_XYZ,
   REG_TYPE_VIRT_VEC3_XYW,
   REG_TYPE_VIRT_VEC3_XZW,
   REG_TYPE_VIRT_VEC3_YZW,
   REG_TYPE_VIRT_VEC2_XY,
   REG_TYPE_VIRT_VEC2_XZ,
   REG_TYPE_VIRT_VEC2_XW,
   REG_TYPE_VIRT_VEC2_YZ,
   REG_TYPE_VIRT_VEC2_YW,
   REG_TYPE_VIRT_VEC2_ZW,
   REG_TYPE_VIRT_SCALAR_X,
   REG_TYPE_VIRT_SCALAR_Y,
   REG_TYPE_VIRT_SCALAR_Z,
   REG_TYPE_VIRT_SCALAR_W,
   REG_TYPE_VIRT_VEC2T_XY,
   REG_TYPE_VIRT_VEC2T_ZW,
   REG_TYPE_VIRT_VEC2C_XY,
   REG_TYPE_VIRT_VEC2C_YZ,
   REG_TYPE_VIRT_VEC2C_ZW,
   REG_TYPE_VIRT_VEC3C_XYZ,
   REG_TYPE_VIRT_VEC3C_YZW,
   NUM_REG_TYPES,
};

/* writemask when used as dest */
static const uint8_t
reg_writemask[NUM_REG_TYPES] = {
   [REG_TYPE_VEC4] = 0xf,
   [REG_TYPE_VIRT_SCALAR_X] = 0x1,
   [REG_TYPE_VIRT_SCALAR_Y] = 0x2,
   [REG_TYPE_VIRT_VEC2_XY] = 0x3,
   [REG_TYPE_VIRT_VEC2T_XY] = 0x3,
   [REG_TYPE_VIRT_VEC2C_XY] = 0x3,
   [REG_TYPE_VIRT_SCALAR_Z] = 0x4,
   [REG_TYPE_VIRT_VEC2_XZ] = 0x5,
   [REG_TYPE_VIRT_VEC2_YZ] = 0x6,
   [REG_TYPE_VIRT_VEC2C_YZ] = 0x6,
   [REG_TYPE_VIRT_VEC3_XYZ] = 0x7,
   [REG_TYPE_VIRT_VEC3C_XYZ] = 0x7,
   [REG_TYPE_VIRT_SCALAR_W] = 0x8,
   [REG_TYPE_VIRT_VEC2_XW] = 0x9,
   [REG_TYPE_VIRT_VEC2_YW] = 0xa,
   [REG_TYPE_VIRT_VEC3_XYW] = 0xb,
   [REG_TYPE_VIRT_VEC2_ZW] = 0xc,
   [REG_TYPE_VIRT_VEC2T_ZW] = 0xc,
   [REG_TYPE_VIRT_VEC2C_ZW] = 0xc,
   [REG_TYPE_VIRT_VEC3_XZW] = 0xd,
   [REG_TYPE_VIRT_VEC3_YZW] = 0xe,
   [REG_TYPE_VIRT_VEC3C_YZW] = 0xe,
};

static inline int reg_get_type(int virt_reg)
{
   return virt_reg % NUM_REG_TYPES;
}

static inline int reg_get_base(struct etna_compile *c, int virt_reg)
{
   /* offset by 1 to avoid reserved position register */
   if (c->nir->info.stage == MESA_SHADER_FRAGMENT)
      return (virt_reg / NUM_REG_TYPES + 1) % ETNA_MAX_TEMPS;
   return virt_reg / NUM_REG_TYPES;
}

struct ra_regs *
etna_ra_setup(void *mem_ctx);

bool
etna_ra_assign(struct etna_compile *c, nir_shader *shader);

unsigned
etna_ra_finish(struct etna_compile *c);

/* verified up to about 65400 instructions (cscap long<n>); 65535 keeps the
 * instruction count and ICACHE_COUNT within 16 bits */
#define ETNA_MAX_SHADER_INSTRUCTIONS (65535)

/* the longest dual16 code to emit (dual16_rewrite()): on GC7000UL r6204,
 * dual16 code hangs the GPU above about 1024 instructions (one launch of
 * straight-line packed half MADs ran at 1011, hung at 1031); 960 keeps a
 * margin. Single-mode code has no such limit. */
#define ETNA_DUAL16_MAX_INSTRUCTIONS 960

static inline void
emit_inst(struct etna_compile *c, struct etna_inst *inst)
{
   if (c->error)
      return;

   if (c->inst_ptr >= (int)c->code_alloc) {
      struct etna_inst *code = NULL;
      unsigned alloc = MIN2(MAX2(c->code_alloc * 2, 1024), ETNA_MAX_SHADER_INSTRUCTIONS);

      if (alloc > c->code_alloc)
         code = realloc(c->code, alloc * sizeof(*code));
      if (!code) {
         mesa_loge("etnaviv: shader longer than %u instructions",
                   ETNA_MAX_SHADER_INSTRUCTIONS);
         c->error = true;
         return;
      }
      c->code = code;
      c->code_alloc = alloc;
   }

   if (c->d16) {
      struct etna_d16_inst *d = &c->d16[c->inst_ptr];

      d->dst = c->d16_dst;
      d->from_nir = c->d16_from_nir;
      for (unsigned s = 0; s < ETNA_NUM_SRC; s++) {
         const struct etna_inst_src *src = &inst->src[s];
         unsigned mask = 0;

         d->src[s] = 0;
         if (!src->use || src->rgroup != ISA_REG_GROUP_TEMP)
            continue;
         for (unsigned j = 0; j < 4; j++)
            mask |= 1 << ((src->swiz >> (2 * j)) & 3);
         for (unsigned r = 0; r < c->d16_nrec; r++) {
            if (c->d16_rec[r].reg == src->reg && (c->d16_rec[r].mask & mask))
               d->src[s] |= c->d16_rec[r].kind;
         }
      }
   }
   c->code[c->inst_ptr++] = *inst;
}

void
etna_emit_alu(struct etna_compile *c, nir_alu_instr *alu, struct etna_inst_dst dst,
              struct etna_inst_src src[3]);

void
etna_emit_tex(struct etna_compile *c, nir_tex_instr *tex, unsigned dst_swiz,
              struct etna_inst_dst dst, struct etna_inst_src coord,
              struct etna_inst_src src1, struct etna_inst_src src2);

void
etna_emit_jump(struct etna_compile *c, unsigned block, struct etna_inst_src condition);

void
etna_emit_discard(struct etna_compile *c, struct etna_inst_src condition);

#endif
