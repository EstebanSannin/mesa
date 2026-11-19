/*
 * Copyright (c) 2012-2019 Etnaviv Project
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
 *    Wladimir J. van der Laan <laanwj@gmail.com>
 */

#include "etna_core_info.h"
#include "etnaviv_compiler.h"
#include "etnaviv_compiler_nir.h"
#include "etnaviv_asm.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_nir.h"
#include "etnaviv_uniforms.h"
#include "etnaviv_util.h"
#include "nir.h"

#include <math.h>
#include "isa/enums.h"
#include "util/u_debug.h"
#include "util/u_memory.h"
#include "util/register_allocate.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_deref.h"
#include "util/format/u_format.h"

#include "util/compiler.h"
#include "util/half_float.h"

static bool
etna_alu_to_scalar_filter_cb(const nir_instr *instr, const void *data)
{
   const struct etna_core_info *info = data;

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   switch (alu->op) {
   case nir_op_frsq:
   case nir_op_frcp:
   case nir_op_flog2:
   case nir_op_fexp2:
   case nir_op_fsqrt:
   case nir_op_fcos:
   case nir_op_fsin:
   case nir_op_fdiv:
   case nir_op_imul_high:
   case nir_op_umul_high:
      return true;
   /* TODO: can do better than alu_to_scalar for vector compares */
   case nir_op_b32all_fequal2:
   case nir_op_b32all_fequal3:
   case nir_op_b32all_fequal4:
   case nir_op_b32any_fnequal2:
   case nir_op_b32any_fnequal3:
   case nir_op_b32any_fnequal4:
   case nir_op_b32all_iequal2:
   case nir_op_b32all_iequal3:
   case nir_op_b32all_iequal4:
   case nir_op_b32any_inequal2:
   case nir_op_b32any_inequal3:
   case nir_op_b32any_inequal4:
      return true;
   case nir_op_fdot2:
      if (!etna_core_has_feature(info, ETNA_FEATURE_HALTI2))
         return true;
      break;
   case nir_op_bitfield_insert:
      /* bit_insert_etna applies one offset and bit count to the whole vector,
       * so scalarize when they vary per component.
       */
      for (unsigned i = 1; i < alu->def.num_components; i++) {
         if (alu->src[2].swizzle[i] != alu->src[2].swizzle[0] ||
             alu->src[3].swizzle[i] != alu->src[3].swizzle[0])
            return true;
      }
      break;
   default:
      break;
   }

   return false;
}

/* IMULLO0 and IMADLO0 are scalar too, but a product is split only after
 * nir_opt_algebraic_late(): it fuses iadd(imul) into an imad where it sees
 * the imul, and a vector product split before it hides behind a vec. An
 * OpenCL int4 multiply-add took four IMULLO0 and a vector ADD instead of
 * four IMADLO0.
 */
static bool
etna_alu_to_scalar_late_filter_cb(const nir_instr *instr, const void *data)
{
   if (instr->type == nir_instr_type_alu) {
      switch (nir_instr_as_alu(instr)->op) {
      case nir_op_imul:
      case nir_op_imad:
         return true;
      default:
         break;
      }
   }

   return etna_alu_to_scalar_filter_cb(instr, data);
}

static void
etna_emit_block_start(struct etna_compile *c, unsigned block)
{
   c->block_ptr[block] = c->inst_ptr;
}

static void
etna_emit_output(struct etna_compile *c, nir_variable *var, struct etna_inst_src src)
{
   struct etna_shader_io_file *sf = &c->variant->outfile;

   if (is_fs(c)) {
      switch (var->data.location) {
      case FRAG_RESULT_DATA0:
      case FRAG_RESULT_DATA1:
      case FRAG_RESULT_DATA2:
      case FRAG_RESULT_DATA3:
      case FRAG_RESULT_DATA4:
      case FRAG_RESULT_DATA5:
      case FRAG_RESULT_DATA6:
      case FRAG_RESULT_DATA7:
         c->variant->ps_color_out_reg[var->data.location - FRAG_RESULT_DATA0] = src.reg;
         break;
      case FRAG_RESULT_DEPTH:
         c->variant->ps_depth_out_reg = src.reg;
         break;
      default:
         UNREACHABLE("Unsupported fs output");
      }
      return;
   }

   switch (var->data.location) {
   case VARYING_SLOT_POS:
      c->variant->vs_pos_out_reg = src.reg;
      break;
   case VARYING_SLOT_PSIZ:
      c->variant->vs_pointsize_out_reg = src.reg;
      break;
   default:
      assert(sf->num_reg < ETNA_NUM_INPUTS);
      sf->reg[sf->num_reg].reg = src.reg;
      sf->reg[sf->num_reg].slot = var->data.location;
      sf->reg[sf->num_reg].num_components = glsl_get_components(var->type);
      sf->num_reg++;
      break;
   }
}

#define OPT(nir, pass, ...) ({                             \
   bool this_progress = false;                             \
   NIR_PASS(this_progress, nir, pass, ##__VA_ARGS__);      \
   this_progress;                                          \
})

static void
etna_optimize_loop(nir_shader *s)
{
   bool progress;
   do {
      progress = false;

      NIR_PASS(_, s, nir_lower_vars_to_ssa);
      progress |= OPT(s, nir_opt_copy_prop_vars);
      progress |= OPT(s, nir_opt_shrink_stores, true);
      progress |= OPT(s, nir_opt_shrink_vectors, false);
      progress |= OPT(s, nir_opt_copy_prop);
      progress |= OPT(s, nir_opt_dce);
      progress |= OPT(s, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 16,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      progress |= OPT(s, nir_opt_peephole_select, &peephole_select_options);
      progress |= OPT(s, nir_opt_intrinsics);
      progress |= OPT(s, nir_opt_algebraic);
      progress |= OPT(s, nir_opt_constant_folding);
      progress |= OPT(s, nir_opt_dead_cf);
      if (OPT(s, nir_opt_loop)) {
         progress = true;
         /* If nir_opt_loop makes progress, then we need to clean
          * things up if we want any hope of nir_opt_if or nir_opt_loop_unroll
          * to make progress.
          */
         OPT(s, nir_opt_copy_prop);
         OPT(s, nir_opt_dce);
      }
      progress |= OPT(s, nir_opt_loop_unroll);
      progress |= OPT(s, nir_opt_if, nir_opt_if_optimize_phi_true_false);
      progress |= OPT(s, nir_opt_remove_phis);
      progress |= OPT(s, nir_opt_undef);
   }
   while (progress);
}

static unsigned
etna_glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
copy_uniform_state_to_shader(struct etna_shader_variant *sobj, uint64_t *consts, unsigned count)
{
   struct etna_shader_uniform_info *uinfo = &sobj->uniforms;

   uinfo->count = count * 4;
   uinfo->data = MALLOC(uinfo->count * sizeof(*uinfo->data));
   uinfo->contents = MALLOC(uinfo->count * sizeof(*uinfo->contents));

   for (unsigned i = 0; i < uinfo->count; i++) {
      uinfo->data[i] = consts[i];
      uinfo->contents[i] = consts[i] >> 32;
   }

   etna_set_shader_uniforms_dirty_flags(sobj);
}

#define ALU_SWIZ(s) INST_SWIZ((s)->swizzle[0], (s)->swizzle[1], (s)->swizzle[2], (s)->swizzle[3])
#define SRC_DISABLE ((hw_src){})
#define SRC_CONST(idx, s) ((hw_src){.use=1, .rgroup = ISA_REG_GROUP_UNIFORM_0, .reg=idx, .swiz=s})
#define SRC_REG(idx, s) ((hw_src){.use=1, .rgroup = ISA_REG_GROUP_TEMP, .reg=idx, .swiz=s})

typedef struct etna_inst_dst hw_dst;
typedef struct etna_inst_src hw_src;

static inline hw_src
src_swizzle(hw_src src, unsigned swizzle)
{
   return etna_src_swizzle(src, swizzle);
}

/* constants are represented as 64-bit ints
 * 32-bit for the value and 32-bit for the type (imm, uniform, etc)
 */

#define CONST_VAL(a, b) (nir_const_value) {.u64 = (uint64_t)(a) << 32 | (uint64_t)(b)}
#define CONST(x) CONST_VAL(ETNA_UNIFORM_CONSTANT, x)
#define UNIFORM(x) CONST_VAL(ETNA_UNIFORM_UNIFORM, x)
#define TEXSCALE(x, i) CONST_VAL(ETNA_UNIFORM_TEXRECT_SCALE_X + (i), x)
#define TEXSIZE(x, i) CONST_VAL(ETNA_UNIFORM_TEXTURE_WIDTH + (i), x)
#define SAMPLERLOD(x, i) CONST_VAL(ETNA_UNIFORM_SAMPLER_LOD_MIN + (i), x)

static int
const_add(uint64_t *c, uint64_t value)
{
   for (unsigned i = 0; i < 4; i++) {
      if (c[i] == value || !c[i]) {
         c[i] = value;
         return i;
      }
   }
   return -1;
}

static inline bool
inline_immediate(uint32_t bits, unsigned *type, uint32_t *imm)
{
   if ((bits & 0xfff) == 0) {
      /* "float" - shifted by 12 */
      *type = 0;
      *imm = bits >> 12;
   } else if (bits < (1 << 20)) {
      /* "unsigned" - raw 20 bit value */
      *type = 2;
      *imm = bits;
   } else if (bits >= 0xfff80000) {
      /* "signed" - sign extended 20-bit (sign included) value */
      *type = 1;
      *imm = bits;
   } else {
      return false;
   }

   return true;
}

static hw_src
const_src(struct etna_compile *c, nir_const_value *value, unsigned num_components)
{
   /* use inline immediates if possible (not in dual16 code) */
   if (c->info->halti >= 2 && num_components == 1 && !c->variant->dual16 &&
       value[0].u64 >> 32 == ETNA_UNIFORM_CONSTANT) {
      unsigned type;
      uint32_t imm;

      if (inline_immediate(value[0].u32, &type, &imm))
         return etna_immediate_src(type, imm);
   }

   unsigned i;
   int swiz = -1;
   for (i = 0; swiz < 0; i++) {
      if (i >= ETNA_MAX_IMM / 4) {
         c->error = true;
         return SRC_CONST(0, INST_SWIZ_IDENTITY);
      }
      uint64_t *a = &c->consts[i*4];
      uint64_t save[4];
      memcpy(save, a, sizeof(save));
      swiz = 0;
      for (unsigned j = 0; j < num_components; j++) {
         int c = const_add(a, value[j].u64);
         if (c < 0) {
            memcpy(a, save, sizeof(save));
            swiz = -1;
            break;
         }
         swiz |= c << j * 2;
      }
   }

   c->const_count = MAX2(c->const_count, i);

   return SRC_CONST(i - 1, swiz);
}

static bool
needs_uniform_slot(const nir_load_const_instr *load_const)
{
   unsigned type;
   uint32_t imm;

   if (load_const->def.bit_size != 32)
      return false;

   return load_const->def.num_components > 1 ||
          !inline_immediate(load_const->value[0].u32, &type, &imm);
}

static bool
has_alu_use(const nir_def *def)
{
   nir_foreach_use(src, def) {
      if (nir_src_use_instr(src)->type == nir_instr_type_alu)
         return true;
   }

   return false;
}

static unsigned
constant_data_add(nir_shader *s, const nir_load_const_instr *load_const)
{
   unsigned size = load_const->def.num_components * 4;
   uint32_t value[4];

   nir_const_value_to_array(value, load_const->value,
                            load_const->def.num_components, u32);

   for (unsigned off = 0; off + size <= s->constant_data_size; off += 4) {
      if (!memcmp((char *)s->constant_data + off, value, size))
         return off;
   }

   unsigned offset = s->constant_data_size;

   s->constant_data_size += size;
   s->constant_data = rerzalloc_size(s, s->constant_data, offset,
                                     s->constant_data_size);
   memcpy((char *)s->constant_data + offset, value, size);

   return offset;
}

static bool
etna_nir_spill_constants(nir_shader *s)
{
   bool progress = false;

   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_load_const)
               continue;

            nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
            nir_def *def = &load_const->def;

            if (!needs_uniform_slot(load_const) || !has_alu_use(def))
               continue;

            unsigned offset = constant_data_add(s, load_const);

            nir_foreach_use_safe(src, def) {
               nir_instr *parent = nir_src_use_instr(src);

               if (parent->type != nir_instr_type_alu)
                  continue;

               b.cursor = nir_before_instr(parent);
               nir_def *load = nir_load_constant(&b, def->num_components, 32,
                                                 nir_imm_int(&b, offset),
                                                 .align_mul = 4);
               nir_src_rewrite(src, load);
               impl_progress = true;
            }
         }
      }

      nir_progress(impl_progress, impl, nir_metadata_control_flow);
      progress |= impl_progress;
   }

   return progress;
}

/* how to swizzle when used as a src */
static const uint8_t
reg_swiz[NUM_REG_TYPES] = {
   [REG_TYPE_VEC4] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_X] = SWIZZLE(X, X, X, X),
   [REG_TYPE_VIRT_SCALAR_Y] = SWIZZLE(Y, Y, Y, Y),
   [REG_TYPE_VIRT_VEC2_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC2T_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC2C_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_Z] = SWIZZLE(Z, Z, Z, Z),
   [REG_TYPE_VIRT_VEC2_XZ] = SWIZZLE(X, Z, X, Z),
   [REG_TYPE_VIRT_VEC2_YZ] = SWIZZLE(Y, Z, Y, Z),
   [REG_TYPE_VIRT_VEC2C_YZ] = SWIZZLE(Y, Z, Y, Z),
   [REG_TYPE_VIRT_VEC3_XYZ] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC3C_XYZ] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_W] = SWIZZLE(W, W, W, W),
   [REG_TYPE_VIRT_VEC2_XW] = SWIZZLE(X, W, X, W),
   [REG_TYPE_VIRT_VEC2_YW] = SWIZZLE(Y, W, Y, W),
   [REG_TYPE_VIRT_VEC3_XYW] = SWIZZLE(X, Y, W, X),
   [REG_TYPE_VIRT_VEC2_ZW] = SWIZZLE(Z, W, Z, W),
   [REG_TYPE_VIRT_VEC2T_ZW] = SWIZZLE(Z, W, Z, W),
   [REG_TYPE_VIRT_VEC2C_ZW] = SWIZZLE(Z, W, Z, W),
   [REG_TYPE_VIRT_VEC3_XZW] = SWIZZLE(X, Z, W, X),
   [REG_TYPE_VIRT_VEC3_YZW] = SWIZZLE(Y, Z, W, X),
   [REG_TYPE_VIRT_VEC3C_YZW] = SWIZZLE(Y, Z, W, X),
};

/* how to swizzle when used as a dest */
static const uint8_t
reg_dst_swiz[NUM_REG_TYPES] = {
   [REG_TYPE_VEC4] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_X] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_Y] = SWIZZLE(X, X, X, X),
   [REG_TYPE_VIRT_VEC2_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC2T_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC2C_XY] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_Z] = SWIZZLE(X, X, X, X),
   [REG_TYPE_VIRT_VEC2_XZ] = SWIZZLE(X, X, Y, Y),
   [REG_TYPE_VIRT_VEC2_YZ] = SWIZZLE(X, X, Y, Y),
   [REG_TYPE_VIRT_VEC2C_YZ] = SWIZZLE(X, X, Y, Y),
   [REG_TYPE_VIRT_VEC3_XYZ] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_VEC3C_XYZ] = INST_SWIZ_IDENTITY,
   [REG_TYPE_VIRT_SCALAR_W] = SWIZZLE(X, X, X, X),
   [REG_TYPE_VIRT_VEC2_XW] = SWIZZLE(X, X, Y, Y),
   [REG_TYPE_VIRT_VEC2_YW] = SWIZZLE(X, X, Y, Y),
   [REG_TYPE_VIRT_VEC3_XYW] = SWIZZLE(X, Y, Z, Z),
   [REG_TYPE_VIRT_VEC2_ZW] = SWIZZLE(X, X, X, Y),
   [REG_TYPE_VIRT_VEC2T_ZW] = SWIZZLE(X, X, X, Y),
   [REG_TYPE_VIRT_VEC2C_ZW] = SWIZZLE(X, X, X, Y),
   [REG_TYPE_VIRT_VEC3_XZW] = SWIZZLE(X, Y, Y, Z),
   [REG_TYPE_VIRT_VEC3_YZW] = SWIZZLE(X, X, Y, Z),
   [REG_TYPE_VIRT_VEC3C_YZW] = SWIZZLE(X, X, Y, Z),
};

/* nir_src to allocated register */
static hw_src
ra_src(struct etna_compile *c, nir_src *src)
{
   unsigned reg = ra_get_node_reg(c->g, c->live_map[src_index(c->impl, src)]);
   return SRC_REG(reg_get_base(c, reg), reg_swiz[reg_get_type(reg)]);
}

static hw_src
get_src(struct etna_compile *c, nir_src *src)
{
   nir_instr *instr = nir_def_instr(src->ssa);

   if (instr->pass_flags & BYPASS_SRC) {
      assert(instr->type == nir_instr_type_alu);
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      assert(alu->op == nir_op_mov);
      return src_swizzle(get_src(c, &alu->src[0].src), ALU_SWIZ(&alu->src[0]));
   }

   switch (instr->type) {
   case nir_instr_type_load_const:
      return const_src(c, nir_instr_as_load_const(instr)->value, src->ssa->num_components);
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_input:
      case nir_intrinsic_load_instance_id:
      case nir_intrinsic_load_vertex_id:
      case nir_intrinsic_load_uniform:
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_global_etna:
      case nir_intrinsic_load_image_etna:
      case nir_intrinsic_global_atomic_etna:
      case nir_intrinsic_global_atomic_swap_etna:
      case nir_intrinsic_load_constant:
      case nir_intrinsic_load_reg:
      case nir_intrinsic_ddx:
      case nir_intrinsic_ddy:
         return ra_src(c, src);
      case nir_intrinsic_load_front_face:
         return (hw_src) { .use = 1, .rgroup = ISA_REG_GROUP_INTERNAL };
      case nir_intrinsic_load_frag_coord:
         return SRC_REG(0, INST_SWIZ_IDENTITY);
      default:
         compile_error(c, "Unhandled NIR intrinsic type: %s\n",
                       nir_intrinsic_infos[intr->intrinsic].name);
         break;
      }
   } break;
   case nir_instr_type_alu:
   case nir_instr_type_tex:
      return ra_src(c, src);
   case nir_instr_type_undef: {
      /* return zero to deal with broken Blur demo */
      nir_const_value value = CONST(0);
      return src_swizzle(const_src(c, &value, 1), SWIZZLE(X,X,X,X));
   }
   default:
      compile_error(c, "Unhandled NIR instruction type: %d\n", instr->type);
      break;
   }

   return SRC_DISABLE;
}

static bool
vec_dest_has_swizzle(nir_alu_instr *vec, nir_def *ssa)
{
   for (unsigned i = 0; i < vec->def.num_components; i++) {
      if (vec->src[i].src.ssa != ssa)
         continue;

      if (vec->src[i].swizzle[0] != i)
         return true;
   }

   /* don't deal with possible bypassed vec/mov chain */
   nir_foreach_use(use_src, ssa) {
      nir_instr *instr = nir_src_use_instr(use_src);
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_mov:
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
         return true;
      default:
         break;
      }
   }
   return false;
}

/* get allocated dest register for nir_def
 * *p_swiz tells how the components need to be placed into register
 */
static hw_dst
ra_def(struct etna_compile *c, nir_def *def, unsigned *p_swiz)
{
   unsigned swiz = INST_SWIZ_IDENTITY, mask = 0xf;
   def = real_def(def, &swiz, &mask);

   unsigned r = ra_get_node_reg(c->g, c->live_map[def_index(c->impl, def)]);
   unsigned t = reg_get_type(r);

   *p_swiz = inst_swiz_compose(swiz, reg_dst_swiz[t]);

   return (hw_dst) {
      .use = 1,
      .reg = reg_get_base(c, r),
      .write_mask = inst_write_mask_compose(mask, reg_writemask[t]),
   };
}

static void
emit_alu(struct etna_compile *c, nir_alu_instr * alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];

   /* marked as dead instruction (vecN and other bypassed instr) */
   if (is_dead_instruction(&alu->instr))
      return;

   assert(!(alu->op >= nir_op_vec2 && alu->op <= nir_op_vec4));

   unsigned dst_swiz;
   hw_dst dst = ra_def(c, &alu->def, &dst_swiz);

   switch (alu->op) {
   case nir_op_fdot2:
   case nir_op_fdot3:
   case nir_op_fdot4:
      /* not per-component - don't compose dst_swiz */
      dst_swiz = INST_SWIZ_IDENTITY;
      break;
   default:
      break;
   }

   hw_src srcs[3] = {0};

   for (int i = 0; i < info->num_inputs; i++) {
      nir_alu_src *asrc = &alu->src[i];
      hw_src src;

      src = src_swizzle(get_src(c, &asrc->src), ALU_SWIZ(asrc));
      src = src_swizzle(src, dst_swiz);

      if (src.rgroup != ISA_REG_GROUP_IMMED) {
         src.neg = is_src_mod_neg(&alu->instr, i) || (alu->op == nir_op_fneg);
         src.abs = is_src_mod_abs(&alu->instr, i) || (alu->op == nir_op_fabs);
      } else {
         assert(alu->op != nir_op_fabs);
         assert(!is_src_mod_abs(&alu->instr, i) && alu->op != nir_op_fabs);

         if (src.imm_type > 0)
            assert(!is_src_mod_neg(&alu->instr, i));

         if (is_src_mod_neg(&alu->instr, i) && src.imm_type == 0)
            src.imm_val ^= 0x80000;
      }

      srcs[i] = src;
   }

   /* src2 is the packed offset and bit count, not per-component data, so keep
    * it instead of composing the destination swizzle onto it. Its own swizzle
    * stays: lower_alu() can merge it with another constant source into one
    * uniform, with offset and bit count in .yz for example.
    */
   if (alu->op == nir_op_bitfield_insert_etna)
      srcs[2] = src_swizzle(src_swizzle(get_src(c, &alu->src[2].src), ALU_SWIZ(&alu->src[2])),
                            SWIZZLE(X, Y, Y, Y));

   etna_emit_alu(c, alu, dst, srcs);
}

static void
emit_tex(struct etna_compile *c, nir_tex_instr * tex)
{
   unsigned dst_swiz;
   hw_dst dst = ra_def(c, &tex->def, &dst_swiz);
   nir_src *coord = NULL, *src1 = NULL, *src2 = NULL;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
         coord = &tex->src[i].src;
         break;
      case nir_tex_src_bias:
      case nir_tex_src_lod:
      case nir_tex_src_ddx:
         assert(!src1);
         src1 = &tex->src[i].src;
         break;
      case nir_tex_src_comparator:
      case nir_tex_src_ddy:
         assert(!src2);
         src2 = &tex->src[i].src;
         break;
      case nir_tex_src_ms_index:
         /* Consumed as an instruction immediate by etna_emit_tex(..). */
         break;
      default:
         compile_error(c, "Unhandled NIR tex src type: %d\n",
                       tex->src[i].src_type);
         break;
      }
   }

   etna_emit_tex(c, tex, dst_swiz, dst, get_src(c, coord),
                 src1 ? get_src(c, src1) : SRC_DISABLE,
                 src2 ? get_src(c, src2) : SRC_DISABLE);
}

/* opcode and type of an atomic; NOP if the hardware has none */
static enum isa_opc
atomic_opcode(nir_atomic_op op, enum isa_type *type)
{
   *type = ISA_TYPE_S32;

   switch (op) {
   case nir_atomic_op_iadd:    return ISA_OPC_ATOMIC_ADD;
   case nir_atomic_op_imin:    return ISA_OPC_ATOMIC_MIN;
   case nir_atomic_op_imax:    return ISA_OPC_ATOMIC_MAX;
   case nir_atomic_op_iand:    return ISA_OPC_ATOMIC_AND;
   case nir_atomic_op_ior:     return ISA_OPC_ATOMIC_OR;
   case nir_atomic_op_ixor:    return ISA_OPC_ATOMIC_XOR;
   case nir_atomic_op_cmpxchg: return ISA_OPC_ATOMIC_CMP_XCHG;
   case nir_atomic_op_umin:    *type = ISA_TYPE_U32; return ISA_OPC_ATOMIC_MIN;
   case nir_atomic_op_umax:    *type = ISA_TYPE_U32; return ISA_OPC_ATOMIC_MAX;
   case nir_atomic_op_xchg:    *type = ISA_TYPE_U32; return ISA_OPC_ATOMIC_XCHG;
   default:                    return ISA_OPC_NOP;
   }
}

/* An image descriptor fills a uniform register, in order: address, stride,
 * size, format word (see etna_image_format_word()). */
static hw_src
image_desc_src(struct etna_compile *c, unsigned image, unsigned plane)
{
   uint64_t want[4];

   for (unsigned i = 0; i < 4; i++)
      want[i] = CONST_VAL(ETNA_UNIFORM_IMAGE_DESC, image | plane << 6 | i << 8).u64;

   for (unsigned r = 0; r < ETNA_MAX_IMM / 4; r++) {
      bool match = true, empty = true;

      for (unsigned i = 0; i < 4; i++) {
         match &= c->consts[r * 4 + i] == want[i];
         empty &= !c->consts[r * 4 + i];
      }

      if (match || empty) {
         memcpy(&c->consts[r * 4], want, sizeof(want));
         c->const_count = MAX2(c->const_count, r + 1);
         return SRC_CONST(r, INST_SWIZ_IDENTITY);
      }
   }

   c->error = true;
   return SRC_CONST(0, INST_SWIZ_IDENTITY);
}

/* img_load/img_store as NXP's compiler emits them for GLES: pack mode, the
 * register type of the format class (f32 for float and normalized formats,
 * which the hardware converts; u32/s32 with .denorm, and .sat for stores, for
 * integer formats), coordinate .xyyy, or .xyzz for the 3D form whose z is the
 * address of the layer. A load writes the components of the format, a store
 * takes .xyzw. Without pack mode and with other fields the instructions work
 * on byte lanes (as for NXP's NPU), one byte per texel. */
static void
emit_image(struct etna_compile *c, nir_intrinsic_instr *intr)
{
   const bool store = intr->intrinsic == nir_intrinsic_store_image_etna;
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool layered = nir_intrinsic_image_array(intr) || dim == GLSL_SAMPLER_DIM_3D ||
                        dim == GLSL_SAMPLER_DIM_CUBE;
   const enum pipe_format format = nir_intrinsic_format(intr);
   const bool sint = util_format_is_pure_sint(format);
   const bool uint = util_format_is_pure_uint(format);

   if (!nir_src_is_const(intr->src[0])) {
      compile_error(c, "dynamically indexed image\n");
      return;
   }

   struct etna_inst inst = {
      .opcode = store ? (layered ? ISA_OPC_IMG_STORE_3D : ISA_OPC_IMG_STORE)
                      : (layered ? ISA_OPC_IMG_LOAD_3D : ISA_OPC_IMG_LOAD),
      .type = sint ? ISA_TYPE_S32 : uint ? ISA_TYPE_U32 : ISA_TYPE_F32,
      .pmode = 1,
      .denorm = sint || uint,
      .sat = store && (sint || uint),
      .src[0] = image_desc_src(c, nir_src_as_uint(intr->src[0]), nir_intrinsic_component(intr)),
      .src[1] = src_swizzle(get_src(c, &intr->src[1]),
                            layered ? SWIZZLE(X, Y, Z, Z) : SWIZZLE(X, Y, Y, Y)),
   };

   if (store) {
      inst.dst.write_mask = ISA_WRMASK_XYZW;
      inst.src[2] = get_src(c, &intr->src[2]);
   } else {
      /* components of the format; 128-bit formats load two per plane,
       * except from buffers */
      const unsigned comps = util_format_get_blocksize(format) == 16 && dim != GLSL_SAMPLER_DIM_BUF
                                ? 2 : util_format_get_nr_components(format);
      unsigned dst_swiz;

      inst.dst = ra_def(c, &intr->def, &dst_swiz);
      inst.dst.write_mask &= BITFIELD_MASK(comps);
   }

   emit_inst(c, &inst);
}

static void
emit_intrinsic(struct etna_compile *c, nir_intrinsic_instr * intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
      etna_emit_output(c, nir_src_as_deref(intr->src[0])->var, get_src(c, &intr->src[1]));
      break;
   case nir_intrinsic_terminate_if:
      etna_emit_discard(c, get_src(c, &intr->src[0]));
      break;
   case nir_intrinsic_terminate:
      etna_emit_discard(c, SRC_DISABLE);
      break;
   case nir_intrinsic_ddx: {
      unsigned dst_swiz;
      struct etna_inst_dst dst = ra_def(c, &intr->def, &dst_swiz);
      struct etna_inst_src src = get_src(c, &intr->src[0]);

      src = src_swizzle(src, dst_swiz);

      struct etna_inst inst = {
         .dst = dst,
         .opcode = ISA_OPC_DSX,
         .cond = ISA_COND_TRUE,
         .type = ISA_TYPE_F32,
         .src[0] = src,
         .src[1] = src,
      };

      emit_inst(c, &inst);
   } break;
   case nir_intrinsic_ddy: {
      unsigned dst_swiz;
      struct etna_inst_dst dst = ra_def(c, &intr->def, &dst_swiz);
      struct etna_inst_src src = get_src(c, &intr->src[0]);

      src = src_swizzle(src, dst_swiz);

      struct etna_inst inst = {
         .dst = dst,
         .opcode = ISA_OPC_DSY,
         .type = ISA_TYPE_F32,
         .cond = ISA_COND_TRUE,
         .src[0] = src,
         .src[1] = src,
      };

      emit_inst(c, &inst);
   } break;
   case nir_intrinsic_load_uniform: {
      unsigned dst_swiz;
      struct etna_inst_dst dst = ra_def(c, &intr->def, &dst_swiz);

      /* TODO: rework so extra MOV isn't required, load up to 4 addresses at once */
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_MOVAR,
         .dst.use = 1,
         .dst.write_mask = ISA_WRMASK_X___,
         .src[0] = get_src(c, &intr->src[0]),
      });
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_MOV,
         .dst = dst,
         .src[0] = {
            .use = 1,
            .rgroup = ISA_REG_GROUP_UNIFORM_0,
            .reg = nir_intrinsic_base(intr),
            .swiz = dst_swiz,
            .amode = ISA_REG_ADDRESSING_MODE_AX,
         },
      });
   } break;
   case nir_intrinsic_load_ubo: {
      /* TODO: if offset is of the form (x + C) then add C to the base instead */
      unsigned idx = nir_src_as_const_value(intr->src[0])[0].u32;
      unsigned dst_swiz;
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_LOAD,
         .type = ISA_TYPE_U32,
         .dst = ra_def(c, &intr->def, &dst_swiz),
         .src[0] = get_src(c, &intr->src[1]),
         .src[1] = const_src(c, &CONST_VAL(ETNA_UNIFORM_UBO_ADDR, idx), 1),
      });
   } break;
   case nir_intrinsic_load_constant: {
      /* The byte offset is a src, so every load shares one address uniform. */
      unsigned dst_swiz;
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_LOAD,
         .type = ISA_TYPE_U32,
         .dst = ra_def(c, &intr->def, &dst_swiz),
         .src[0] = get_src(c, &intr->src[0]),
         .src[1] = const_src(c, &CONST_VAL(ETNA_UNIFORM_CONSTANT_DATA_ADDR, nir_intrinsic_base(intr)), 1),
      });
   } break;
   case nir_intrinsic_store_global_offset:
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_STORE,
         .type = ISA_TYPE_U32,
         .denorm = 1,
         .dst = {
            .write_mask = nir_intrinsic_write_mask(intr),
         },
         .src[0] = get_src(c, &intr->src[1]),
         .src[1] = get_src(c, &intr->src[2]),
         .src[2] = get_src(c, &intr->src[0]),
      });
      break;
   case nir_intrinsic_load_image_etna:
   case nir_intrinsic_store_image_etna:
      emit_image(c, intr);
      break;
   case nir_intrinsic_load_global_etna: {
      unsigned dst_swiz;
      const nir_alu_type type = nir_intrinsic_dest_type(intr);
      /* the base first, as NXP's compiler: LEFT_SHIFT applies to the second
       * operand, which can also be an immediate */
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_LOAD,
         .type = type == nir_type_uint8 ? ISA_TYPE_U8 : type == nir_type_uint16 ? ISA_TYPE_U16 : ISA_TYPE_U32,
         .left_shift = nir_intrinsic_offset_shift(intr),
         .dst = ra_def(c, &intr->def, &dst_swiz),
         .src[0] = get_src(c, &intr->src[0]),
         .src[1] = get_src(c, &intr->src[1]),
      });
   } break;
   case nir_intrinsic_store_global_etna:
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_STORE,
         .type = nir_intrinsic_src_type(intr) == nir_type_uint8 ? ISA_TYPE_U8 :
                 nir_intrinsic_src_type(intr) == nir_type_uint16 ? ISA_TYPE_U16 :
                 nir_intrinsic_src_type(intr) == nir_type_float16 ? ISA_TYPE_F16 : ISA_TYPE_U32,
         .denorm = 1,
         .left_shift = nir_intrinsic_offset_shift(intr),
         .dst = {
            .write_mask = nir_intrinsic_write_mask(intr),
         },
         .src[0] = get_src(c, &intr->src[1]),
         .src[1] = get_src(c, &intr->src[2]),
         .src[2] = get_src(c, &intr->src[0]),
      });
      break;
   case nir_intrinsic_global_atomic_etna:
   case nir_intrinsic_global_atomic_swap_etna: {
      enum isa_type type;
      enum isa_opc opc = atomic_opcode(nir_intrinsic_atomic_op(intr), &type);
      unsigned dst_swiz;

      if (opc == ISA_OPC_NOP) {
         compile_error(c, "unsupported atomic op %d\n", nir_intrinsic_atomic_op(intr));
         break;
      }

      struct etna_inst_src data = get_src(c, &intr->src[2]);
      struct etna_inst_dst dst = ra_def(c, &intr->def, &dst_swiz);

      /* atomic_cmp_xchg takes the new and the compared value from .x and .y of
       * its destination register, like the blob emits it (dst t0.x, data
       * t0.xy). The result has a full register that no source of this
       * instruction shares (see RA): copy the operand there and let the old
       * value replace .x. The operand itself can stay live, in a loop. */
      if (opc == ISA_OPC_ATOMIC_CMP_XCHG) {
         emit_inst(c, &(struct etna_inst) {
            .opcode = ISA_OPC_MOV,
            .dst = { .use = 1, .reg = dst.reg, .write_mask = ISA_WRMASK_XY__ },
            .src[0] = data,
         });
         emit_inst(c, &(struct etna_inst) {
            .opcode = opc,
            .type = type,
            .skphp = 1,
            .dst = { .use = 1, .reg = dst.reg, .write_mask = ISA_WRMASK_X___ },
            .src[0] = get_src(c, &intr->src[0]),
            .src[1] = get_src(c, &intr->src[1]),
            .src[2] = SRC_REG(dst.reg, SWIZZLE(X, Y, Y, Y)),
         });
         break;
      }

      /* like the blob: address, offset, data */
      emit_inst(c, &(struct etna_inst) {
         .opcode = opc,
         .type = type,
         .skphp = 1,
         .dst = dst,
         .src[0] = get_src(c, &intr->src[0]),
         .src[1] = get_src(c, &intr->src[1]),
         .src[2] = data,
      });
   } break;
   case nir_intrinsic_barrier:
      if (nir_intrinsic_execution_scope(intr) >= SCOPE_WORKGROUP)
         emit_inst(c, &(struct etna_inst) { .opcode = ISA_OPC_BARRIER });
      break;
   case nir_intrinsic_load_front_face:
   case nir_intrinsic_load_frag_coord:
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_decl_reg:
   case nir_intrinsic_load_reg:
   case nir_intrinsic_store_reg:
      break;
   default:
      compile_error(c, "Unhandled NIR intrinsic type: %s\n",
                    nir_intrinsic_infos[intr->intrinsic].name);
   }
}

/* dual16: a 16-bit value, or a 32-bit copy of one (insert_vec_mov()), is
 * packed halves */
static bool
dual16_packed_def(nir_def *def)
{
   if (def->bit_size == 16)
      return true;

   nir_instr *instr = nir_def_instr(def);
   return instr->type == nir_instr_type_alu && nir_instr_as_alu(instr)->op == nir_op_mov &&
          nir_instr_as_alu(instr)->src[0].src.ssa->bit_size == 16;
}

static bool
dual16_record_src(nir_src *src, void *data)
{
   struct etna_compile *c = data;
   nir_instr *parent = nir_def_instr(src->ssa);
   bool temp = parent->type == nir_instr_type_alu;

   if (parent->type == nir_instr_type_intrinsic) {
      const nir_intrinsic_op op = nir_instr_as_intrinsic(parent)->intrinsic;
      temp = op == nir_intrinsic_load_input || op == nir_intrinsic_load_reg;
   }
   if (!temp || c->d16_nrec >= ARRAY_SIZE(c->d16_rec))
      return true;

   hw_src h = get_src(c, src);
   if (!h.use || h.rgroup != ISA_REG_GROUP_TEMP)
      return true;

   unsigned mask = 0;
   for (unsigned j = 0; j < 4; j++)
      mask |= 1 << ((h.swiz >> (2 * j)) & 3);
   c->d16_rec[c->d16_nrec].reg = h.reg;
   c->d16_rec[c->d16_nrec].mask = mask;
   c->d16_rec[c->d16_nrec].kind = dual16_packed_def(src->ssa) ? ETNA_D16_P : ETNA_D16_W;
   c->d16_nrec++;
   return true;
}

/* dual16: the kinds of the result and the register sources of the
 * instructions emitted for instr (see emit_inst()) */
static void
dual16_record(struct etna_compile *c, nir_instr *instr)
{
   c->d16_nrec = 0;
   c->d16_dst = ETNA_D16_W;
   c->d16_from_nir = false;

   /* a vecN or a modifier folded into its users emits nothing, and the
    * values written straight into a vecN's register have none of their own */
   if (instr->type == nir_instr_type_alu && is_dead_instruction(instr))
      return;

   if (instr->type == nir_instr_type_alu) {
      c->d16_dst = dual16_packed_def(&nir_instr_as_alu(instr)->def) ? ETNA_D16_P : ETNA_D16_W;
      c->d16_from_nir = true;
   } else if (instr->type == nir_instr_type_intrinsic &&
              nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global_etna) {
      c->d16_from_nir = true;
   }
   if (c->d16_from_nir)
      nir_foreach_src(instr, dual16_record_src, c);
}

static void
emit_instr(struct etna_compile *c, nir_instr * instr)
{
   if (c->d16)
      dual16_record(c, instr);
   switch (instr->type) {
   case nir_instr_type_alu:
      emit_alu(c, nir_instr_as_alu(instr));
      break;
   case nir_instr_type_tex:
      emit_tex(c, nir_instr_as_tex(instr));
      break;
   case nir_instr_type_intrinsic:
      emit_intrinsic(c, nir_instr_as_intrinsic(instr));
      break;
   case nir_instr_type_jump:
      assert(nir_instr_is_last(instr));
      break;
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_deref:
      break;
   default:
      compile_error(c, "Unhandled NIR instruction type: %d\n", instr->type);
      break;
   }
   c->d16_from_nir = false;
   c->d16_nrec = 0;
}

static void
emit_block(struct etna_compile *c, nir_block * block)
{
   etna_emit_block_start(c, block->index);

   nir_foreach_instr(instr, block)
      emit_instr(c, instr);

   /* succs->index < block->index is for the loop case  */
   nir_block *succs = block->successors[0];
   if (nir_block_ends_in_jump(block) || succs->index < block->index)
      etna_emit_jump(c, succs->index, SRC_DISABLE);
}

static void
emit_cf_list(struct etna_compile *c, struct exec_list *list);

static void
emit_if(struct etna_compile *c, nir_if * nif)
{
   etna_emit_jump(c, nir_if_first_else_block(nif)->index, get_src(c, &nif->condition));
   emit_cf_list(c, &nif->then_list);

   /* jump at end of then_list to skip else_list
    * not needed if then_list already ends with a jump or else_list is empty
    */
   if (!nir_block_ends_in_jump(nir_if_last_then_block(nif)) &&
       !nir_cf_list_is_empty_block(&nif->else_list))
      etna_emit_jump(c, nir_if_last_then_block(nif)->successors[0]->index, SRC_DISABLE);

   emit_cf_list(c, &nif->else_list);
}

static void
emit_cf_list(struct etna_compile *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         emit_block(c, nir_cf_node_as_block(node));
         break;
      case nir_cf_node_if:
         emit_if(c, nir_cf_node_as_if(node));
         break;
      case nir_cf_node_loop:
         assert(!nir_loop_has_continue_construct(nir_cf_node_as_loop(node)));
         emit_cf_list(c, &nir_cf_node_as_loop(node)->body);
         break;
      default:
         compile_error(c, "Unknown NIR node type\n");
         break;
      }
   }
}

/* based on nir_lower_vec_to_movs */
static unsigned
insert_vec_mov(nir_alu_instr *vec, unsigned start_idx, nir_shader *shader)
{
   assert(start_idx < nir_op_infos[vec->op].num_inputs);
   unsigned write_mask = (1u << start_idx);

   nir_alu_instr *mov = nir_alu_instr_create(shader, nir_op_mov);
   nir_alu_src_copy(&mov->src[0], &vec->src[start_idx]);

   mov->src[0].swizzle[0] = vec->src[start_idx].swizzle[0];

   if (is_src_mod_neg(&vec->instr, start_idx))
      set_src_mod_neg(&mov->instr, 0);

   if (is_src_mod_abs(&vec->instr, start_idx))
      set_src_mod_abs(&mov->instr, 0);

   unsigned num_components = 1;

   for (unsigned i = start_idx + 1; i < vec->def.num_components; i++) {
      if (nir_srcs_equal(vec->src[i].src, vec->src[start_idx].src) &&
         is_src_mod_neg(&vec->instr, i) == is_src_mod_neg(&vec->instr, start_idx) &&
         is_src_mod_abs(&vec->instr, i) == is_src_mod_neg(&vec->instr, start_idx)) {
         write_mask |= (1 << i);
         mov->src[0].swizzle[num_components] = vec->src[i].swizzle[0];
         num_components++;
      }
   }

   nir_def_init(&mov->instr, &mov->def, num_components, 32);

   /* replace vec srcs with inserted mov */
   for (unsigned i = 0, j = 0; i < 4; i++) {
      if (!(write_mask & (1 << i)))
         continue;

      nir_src_rewrite(&vec->src[i].src, &mov->def);
      vec->src[i].swizzle[0] = j++;
   }

   nir_instr_insert_before(&vec->instr, &mov->instr);

   return write_mask;
}

/*
 * Get the nir_const_value from an alu src.  Also look at
 * the parent instruction as it could be a fabs/fneg.
 */
static nir_const_value *get_alu_cv(nir_alu_src *src)
 {
   nir_const_value *cv = nir_src_as_const_value(src->src);

   if (!cv &&
       (nir_src_is_alu(src->src))) {
      nir_alu_instr *parent = nir_def_as_alu(src->src.ssa);

      if ((parent->op == nir_op_fabs) ||
          (parent->op == nir_op_fneg)) {
         cv = nir_src_as_const_value(parent->src[0].src);

         if (cv) {
            /* Validate that we are only using ETNA_UNIFORM_CONSTANT const_values. */
            for (unsigned i = 0; i < parent->def.num_components; i++) {
               if (cv[i].u64 >> 32 != ETNA_UNIFORM_CONSTANT) {
                  cv = NULL;
                  break;
               }
            }
         }
      }
   }

   return cv;
 }

/*
 * for vecN instructions:
 * -merge constant sources into a single src
 * -insert movs (nir_lower_vec_to_movs equivalent)
 * for non-vecN instructions:
 * -try to merge constants as single constant
 * -insert movs for multiple constants if required
 */
static void
lower_alu(struct etna_compile *c, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];

   nir_builder b = nir_builder_at(nir_before_instr(&alu->instr));

   switch (alu->op) {
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
      break;
   default:
      if (etna_core_has_feature(c->info, ETNA_FEATURE_SH_NO_ONECONST_LIMIT))
         return;

      nir_const_value value[4] = {};
      uint8_t swizzle[4][4] = {};
      unsigned swiz_max = 0, num_different_const_srcs = 0;
      int first_const = -1;

      for (unsigned i = 0; i < info->num_inputs; i++) {
         nir_const_value *cv = get_alu_cv(&alu->src[i]);
         if (!cv)
            continue;

         unsigned num_components = info->input_sizes[i] ?: alu->def.num_components;
         for (unsigned j = 0; j < num_components; j++) {
            int idx = const_add(&value[0].u64, cv[alu->src[i].swizzle[j]].u64);
            swizzle[i][j] = idx;
            swiz_max = MAX2(swiz_max, (unsigned) idx);
         }

         if (first_const == -1)
            first_const = i;

         if (!nir_srcs_equal(alu->src[first_const].src, alu->src[i].src))
            num_different_const_srcs++;
      }

      /* nothing to do */
      if (num_different_const_srcs == 0)
         return;

      /* resolve with single combined const src */
      if (swiz_max < 4) {
         nir_def *def = nir_build_imm(&b, swiz_max + 1, 32, value);

         for (unsigned i = 0; i < info->num_inputs; i++) {
            nir_const_value *cv = get_alu_cv(&alu->src[i]);
            if (!cv)
               continue;

            nir_src_rewrite(&alu->src[i].src, def);

            for (unsigned j = 0; j < 4; j++)
               alu->src[i].swizzle[j] = swizzle[i][j];
         }
         return;
      }

      /* resolve with movs */
      unsigned num_const = 0;
      for (unsigned i = 0; i < info->num_inputs; i++) {
         nir_const_value *cv = get_alu_cv(&alu->src[i]);
         if (!cv)
            continue;

         num_const++;
         if (num_const == 1)
            continue;

         nir_def *mov = nir_mov(&b, alu->src[i].src.ssa);
         nir_src_rewrite(&alu->src[i].src, mov);
      }
      return;
   }

   nir_const_value value[4];
   unsigned num_components = 0;

   for (unsigned i = 0; i < info->num_inputs; i++) {
      nir_const_value *cv = get_alu_cv(&alu->src[i]);
      if (cv)
         value[num_components++] = cv[alu->src[i].swizzle[0]];
   }

   /* if there is more than one constant source to the vecN, combine them
    * into a single load_const (removing the vecN completely if all components
    * are constant)
    */
   if (num_components > 1) {
      nir_def *def = nir_build_imm(&b, num_components, 32, value);

      if (num_components == info->num_inputs) {
         nir_def_replace(&alu->def, def);
         return;
      }

      for (unsigned i = 0, j = 0; i < info->num_inputs; i++) {
         nir_const_value *cv = get_alu_cv(&alu->src[i]);
         if (!cv)
            continue;

         nir_src_rewrite(&alu->src[i].src, def);
         alu->src[i].swizzle[0] = j++;
      }
   }

   unsigned finished_write_mask = 0;
   for (unsigned i = 0; i < alu->def.num_components; i++) {
      nir_def *ssa = alu->src[i].src.ssa;

      /* check that vecN instruction is only user of this */
      bool need_mov = false;
      nir_foreach_use_including_if(use_src, ssa) {
         if (nir_src_is_if(use_src) || nir_src_use_instr(use_src) != &alu->instr)
            need_mov = true;
      }

      nir_instr *instr = nir_def_instr(ssa);
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_tex:
         break;
      case nir_instr_type_intrinsic:
         if (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_input) {
            need_mov = vec_dest_has_swizzle(alu, &nir_instr_as_intrinsic(instr)->def);
            break;
         }
         FALLTHROUGH;
      default:
         need_mov = true;
      }

      if (need_mov && !(finished_write_mask & (1 << i)))
         finished_write_mask |= insert_vec_mov(alu, i, c->nir);
   }
}

static bool
emit_shader(struct etna_compile *c, unsigned *num_temps, unsigned *num_consts)
{
   nir_shader *shader = c->nir;
   c->impl = nir_shader_get_entrypoint(shader);

   bool have_indirect_uniform = false;
   unsigned indirect_max = 0;

   nir_builder b = nir_builder_create(c->impl);

   /* convert non-dynamic uniform loads to constants, etc */
   nir_foreach_block(block, c->impl) {
      nir_foreach_instr_safe(instr, block) {
         switch(instr->type) {
         case nir_instr_type_alu:
            /* deals with vecN and const srcs */
            lower_alu(c, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_load_const: {
            nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
            /* dual16: a half constant goes up as a float, which packed
             * instructions read as a half */
            if (c->variant->dual16 && load_const->def.bit_size == 16) {
               for (unsigned i = 0; i < load_const->def.num_components; i++)
                  load_const->value[i].u64 = fui(_mesa_half_to_float(load_const->value[i].u16));
            }
            for (unsigned  i = 0; i < load_const->def.num_components; i++)
               load_const->value[i] = CONST(load_const->value[i].u32);
         } break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            nir_const_value value[4];

            /* TODO: load_ubo can also become a constant in some cases
             * (at the moment it can end up emitting a LOAD with two
             *  uniform sources, which could be a problem on HALTI2)
             */
            switch (intr->intrinsic) {
            case nir_intrinsic_load_uniform: {
               nir_const_value *off = nir_src_as_const_value(intr->src[0]);
               if (!off || off[0].u64 >> 32 != ETNA_UNIFORM_CONSTANT) {
                  have_indirect_uniform = true;
                  indirect_max = nir_intrinsic_base(intr) +
                                 nir_intrinsic_range(intr);
                  continue;
               }

               unsigned base = nir_intrinsic_base(intr);
               /* pre halti2 uniform offset will be float */
               if (c->info->halti < 2)
                  base += (unsigned) off[0].f32;
               else
                  base += off[0].u32;

               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = UNIFORM(base * 4 + i);
            } break;
            case nir_intrinsic_load_texture_scale: {
               int sampler = nir_src_as_int(intr->src[0]);

               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = TEXSCALE(sampler, i);
            } break;
            case nir_intrinsic_load_texture_size_etna: {
               int sampler = nir_src_as_int(intr->src[0]);

               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = TEXSIZE(sampler, i);
            } break;
            case nir_intrinsic_load_sampler_lod_parameters: {
               int sampler = nir_src_as_int(intr->src[0]);

               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = SAMPLERLOD(sampler, i);
            } break;
            case nir_intrinsic_load_xfb_address:
               value[0] = CONST_VAL(ETNA_UNIFORM_XFB_ADDR, nir_intrinsic_base(intr));
               break;
            case nir_intrinsic_load_num_vertices:
               value[0] = CONST_VAL(ETNA_UNIFORM_XFB_NUM_VERTICES, 0);
               break;
            case nir_intrinsic_load_first_vertex:
               value[0] = CONST_VAL(ETNA_UNIFORM_XFB_FIRST_VERTEX, 0);
               break;
            case nir_intrinsic_load_workgroup_size:
               for (unsigned i = 0; i < 3; i++)
                  value[i] = CONST_VAL(ETNA_UNIFORM_WORKGROUP_SIZE, i);
               break;
            case nir_intrinsic_load_num_workgroups:
               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = CONST_VAL(ETNA_UNIFORM_NUM_WORKGROUPS, i);
               break;
            case nir_intrinsic_load_base_workgroup_id:
               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = CONST_VAL(ETNA_UNIFORM_BASE_WORKGROUP_ID, i);
               break;
            case nir_intrinsic_get_ssbo_size:
               value[0] = CONST_VAL(ETNA_UNIFORM_SSBO_SIZE, nir_src_as_uint(intr->src[0]));
               break;
            case nir_intrinsic_load_scratch_base_ptr:
               /* BASE 1: OpenCL private memory, BASE 0: workgroup folding */
               value[0] = CONST_VAL(nir_intrinsic_base(intr) ? ETNA_UNIFORM_PRIVATE_ADDR :
                                                               ETNA_UNIFORM_SCRATCH_ADDR, 0);
               break;
            case nir_intrinsic_load_shared_slice_etna:
               value[0] = CONST_VAL(ETNA_UNIFORM_SHARED_SLICE, 0);
               break;
            case nir_intrinsic_load_shared_base_ptr:
               value[0] = CONST_VAL(ETNA_UNIFORM_SHARED_ADDR, 0);
               break;
            case nir_intrinsic_load_image_param_etna:
               if (!nir_src_is_const(intr->src[0])) {
                  compile_error(c, "dynamically indexed image\n");
                  continue;
               }
               value[0] = CONST_VAL(ETNA_UNIFORM_IMAGE_ADDR + nir_intrinsic_base(intr),
                                    nir_src_as_uint(intr->src[0]));
               break;
            case nir_intrinsic_load_ssbo_address_etna:
               if (!nir_src_is_const(intr->src[0])) {
                  compile_error(c, "dynamically indexed SSBO\n");
                  continue;
               }
               value[0] = CONST_VAL(ETNA_UNIFORM_SSBO_ADDR, nir_src_as_uint(intr->src[0]));
               break;
            default:
               continue;
            }

            b.cursor = nir_after_instr(instr);
            nir_def *def = nir_build_imm(&b, intr->def.num_components, 32, value);

            nir_def_rewrite_uses(&intr->def, def);
            nir_instr_remove(instr);
         } break;
         default:
            break;
         }
      }
   }

   /* TODO: only emit required indirect uniform ranges */
   if (have_indirect_uniform) {
      for (unsigned i = 0; i < indirect_max * 4; i++)
         c->consts[i] = UNIFORM(i).u64;
      c->const_count = indirect_max;
   }

   /* add mov for any store output using sysval/const and for depth stores from intrinsics */
   nir_foreach_block(block, c->impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

         switch (intr->intrinsic) {
         case nir_intrinsic_store_deref: {
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            nir_src *src = &intr->src[1];
            if (nir_src_is_const(*src) || is_sysval(nir_def_instr(src->ssa)) ||
                (shader->info.stage == MESA_SHADER_FRAGMENT &&
                 deref->var->data.location == FRAG_RESULT_DEPTH &&
                 !nir_def_is_alu(src->ssa))) {
               b.cursor = nir_before_instr(instr);
               nir_src_rewrite(src, nir_mov(&b, src->ssa));
            }
         } break;
         case nir_intrinsic_load_image_etna:
         case nir_intrinsic_store_image_etna:
            /* the descriptor is a uniform: coordinate and value go through
             * registers (src 0 is the image index) */
            for (unsigned i = 1; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; i++) {
               if (nir_src_is_const(intr->src[i])) {
                  b.cursor = nir_before_instr(instr);
                  nir_src_rewrite(&intr->src[i], nir_mov(&b, intr->src[i].ssa));
               }
            }
            break;
         case nir_intrinsic_load_global_etna:
         case nir_intrinsic_store_global_etna:
         case nir_intrinsic_global_atomic_etna:
         case nir_intrinsic_global_atomic_swap_etna: {
            /* the address is a uniform: offset and data that would be
             * another uniform go through registers (one distinct uniform per
             * instruction). An inline immediate is not a uniform, and LOAD
             * takes one as its offset (NXP's code has load u0.x, 0:s20);
             * STORE with an immediate offset or value faulted (MMU, OpenCV's
             * Canny), so its constants stay in registers */
            const bool store = intr->intrinsic == nir_intrinsic_store_global_etna;
            const bool load = intr->intrinsic == nir_intrinsic_load_global_etna;
            nir_src *srcs[] = { store ? &intr->src[2] : &intr->src[1],
                                store ? &intr->src[0] : load ? NULL : &intr->src[2] };

            for (unsigned i = 0; i < ARRAY_SIZE(srcs); i++) {
               if (srcs[i] && nir_src_is_const(*srcs[i]) && load &&
                   !needs_uniform_slot(nir_instr_as_load_const(nir_def_instr(srcs[i]->ssa))))
                  continue;
               if (srcs[i] && nir_src_is_const(*srcs[i])) {
                  b.cursor = nir_before_instr(instr);
                  nir_src_rewrite(srcs[i], nir_mov(&b, srcs[i]->ssa));
               }
            }
         } break;
         default:
            break;
         }
      }
   }

   /* An input or a system value sits in a fixed register and emits no
    * instruction. As a phi source, nir_convert_from_ssa() can merge it into
    * the phi's register web, which then gets a single fixed register: with
    * two inputs feeding one phi, one of them is read from the other's
    * register, and the first input's register is free for other values.
    * Copy such sources in the predecessor block instead.
    */
   nir_foreach_block(block, c->impl) {
      nir_foreach_phi(phi, block) {
         nir_foreach_phi_src(src, phi) {
            nir_instr *instr = nir_def_instr(src->src.ssa);
            bool fixed = is_sysval(instr);

            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               fixed |= op == nir_intrinsic_load_input ||
                        op == nir_intrinsic_load_instance_id ||
                        op == nir_intrinsic_load_vertex_id;
            }
            if (!fixed)
               continue;

            b.cursor = nir_after_block_before_jump(src->pred);
            nir_src_rewrite(&src->src, nir_mov(&b, src->src.ssa));
         }
      }
   }

   /* the backend has 32-bit registers only (and 1-bit booleans) */
   nir_foreach_block(block, c->impl) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (def && (def->bit_size == 8 || (def->bit_size == 16 && !c->variant->dual16)) &&
             instr->type != nir_instr_type_load_const) {
            char *str = nir_instr_as_str(instr, NULL);
            compile_error(c, "%u-bit value not lowered: %s\n", def->bit_size, str);
            ralloc_free(str);
            return false;
         }
      }
   }

   /* call directly to avoid validation (load_const don't pass validation at this point) */
   nir_convert_from_ssa(shader, true, false);
   nir_trivialize_registers(shader);

   if (!etna_ra_assign(c, shader)) {
      mesa_loge("etnaviv: register allocation failed (%u nodes)", c->num_nodes);
      ralloc_free(c->g);
      ralloc_free(c->live_map);
      return false;
   }

   emit_cf_list(c, &nir_shader_get_entrypoint(shader)->body);

   *num_temps = etna_ra_finish(c);
   *num_consts = c->const_count;
   return !c->error;
}

static inline unsigned
max_uniforms(const struct etna_shader_variant *v)
{
   const struct etna_specs *specs = v->shader->specs;

   return v->stage == MESA_SHADER_VERTEX ? specs->max_vs_uniforms
                                         : specs->max_ps_uniforms;
}

static bool
etna_compile_check_limits(struct etna_shader_variant *v)
{
   const struct etna_core_info *info = v->shader->info;
   const struct etna_specs *specs = v->shader->specs;

   if (!specs->has_icache && v->needs_icache) {
      DBG("Number of instructions (%d) exceeds maximum %d", v->code_size / 4,
          specs->max_instructions);
      return false;
   }

   /* The NUM_TEMPS fields of VS/PS_TEMP_REGISTER_CONTROL have 6 bits: a
    * shader that takes all 64 temporaries would program 0 and run with
    * garbage registers (MMU faults, GPU hangs). */
   const unsigned max_temps = MIN2(info->gpu.max_registers,
                                   VIVS_PS_TEMP_REGISTER_CONTROL_NUM_TEMPS__MASK);

   if (v->num_temps > max_temps) {
      DBG("Number of registers (%d) exceeds maximum %d", v->num_temps, max_temps);
      return false;
   }

   if (v->uniforms.count / 4 > max_uniforms(v)) {
      DBG("Number of uniforms (%d) exceeds maximum %d",
          v->uniforms.count / 4, max_uniforms(v));
      return false;
   }

   if (v->stage == MESA_SHADER_VERTEX) {
      int num_outputs = v->vs_pointsize_out_reg >= 0 ? 2 : 1;

      num_outputs += v->outfile.num_reg;

      if (num_outputs > specs->max_vs_outputs) {
         DBG("Number of VS outputs (%zu) exceeds maximum %d",
             v->outfile.num_reg, specs->max_vs_outputs);
         return false;
      }
   }

   return true;
}

static void
fill_vs_mystery(struct etna_shader_variant *v)
{
   const struct etna_core_info *info = v->shader->info;

   v->input_count_unk8 = DIV_ROUND_UP(v->infile.num_reg + 4, 16); /* XXX what is this */

   /* fill in "mystery meat" load balancing value. This value determines how
    * work is scheduled between VS and PS
    * in the unified shader architecture. More precisely, it is determined from
    * the number of VS outputs, as well as chip-specific
    * vertex output buffer size, vertex cache size, and the number of shader
    * cores.
    *
    * XXX this is a conservative estimate, the "optimal" value is only known for
    * sure at link time because some
    * outputs may be unused and thus unmapped. Then again, in the general use
    * case with GLSL the vertex and fragment
    * shaders are linked already before submitting to Gallium, thus all outputs
    * are used.
    *
    * note: TGSI compiler counts all outputs (including position and pointsize), here
    * v->outfile.num_reg only counts varyings, +1 to compensate for the position output
    * TODO: might have a problem that we don't count pointsize when it is used
    */

   int half_out = v->outfile.num_reg / 2 + 1;
   assert(half_out);

   uint32_t b = ((20480 / (info->gpu.vertex_output_buffer_size -
                           2 * half_out * info->gpu.vertex_cache_size)) +
                 9) /
                10;
   uint32_t a = (b + 256 / (info->gpu.shader_core_count * half_out)) / 2;
   v->vs_load_balancing = VIVS_VS_LOAD_BALANCING_A(MIN2(a, 255)) |
                             VIVS_VS_LOAD_BALANCING_B(MIN2(b, 255)) |
                             VIVS_VS_LOAD_BALANCING_C(0x3f) |
                             VIVS_VS_LOAD_BALANCING_D(0x0f);
}

/* NIR's fmin and fmax return the other operand for a NaN one (minNum and
 * maxNum) when they preserve NaNs, as OpenCL's do. A SELECT with a
 * comparison, as they are emitted, does that for a NaN second operand but
 * returns a NaN first one: test the first one before. */
static bool
lower_fminmax_nan(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       !nir_alu_instr_is_nan_preserve(alu) || alu->def.bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *y = nir_ssa_for_alu_src(b, alu, 1);

   const unsigned save_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = 0;
   nir_def *sel = alu->op == nir_op_fmax ? nir_fmax(b, x, y) : nir_fmin(b, x, y);
   b->fp_math_ctrl = save_math_ctrl;

   /* the bits, not a comparison: x is NaN when |x| is above infinity */
   nir_def *x_nan = nir_ult(b, nir_imm_int(b, 0x7f800000), nir_iand_imm(b, x, 0x7fffffff));
   nir_def_replace(&alu->def, nir_bcsel(b, x_nan, y, sel));
   return true;
}

/* CL_CONFIG value order 3: t0 = global ID, t1 = local ID, t2 = workgroup ID */
static nir_def *
load_cs_input(nir_builder *b, unsigned reg, unsigned *input_count)
{
   *input_count = MAX2(*input_count, reg + 1);
   return nir_load_input(b, 3, 32, nir_imm_int(b, 0), .base = reg);
}

static bool
lower_cs_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned *input_count = data;
   nir_def *val;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_invocation_id:
      val = load_cs_input(b, 0, input_count);
      break;
   case nir_intrinsic_load_local_invocation_id:
      val = load_cs_input(b, 1, input_count);
      break;
   case nir_intrinsic_load_workgroup_id:
      /* the hardware counts the workgroups of each dispatch from 0, and
       * etna_launch_grid() may dispatch a grid in parts: the first workgroup
       * of the part is a uniform */
      val = nir_iadd(b, load_cs_input(b, 2, input_count), nir_load_base_workgroup_id(b, 32));
      break;
   case nir_intrinsic_load_local_invocation_index: {
      nir_def *id = load_cs_input(b, 1, input_count);
      nir_def *size;

      /* OpenCL kernels have the workgroup size of the dispatch */
      if (b->shader->info.workgroup_size_variable) {
         size = nir_load_workgroup_size(b);
      } else {
         const uint16_t *s = b->shader->info.workgroup_size;
         size = nir_imm_ivec3(b, s[0], s[1], s[2]);
      }

      val = nir_iadd(b, nir_channel(b, id, 0),
                     nir_imul(b, nir_iadd(b, nir_channel(b, id, 1),
                                          nir_imul(b, nir_channel(b, id, 2), nir_channel(b, size, 1))),
                              nir_channel(b, size, 0)));
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, val, intr->def.bit_size));
   return true;
}

/* a narrow store: STORE.u8/u16 takes a 32-bit register and writes its low
 * byte or half */
static nir_def *
store_value32(nir_builder *b, nir_def *v)
{
   return v->bit_size < 32 ? nir_u2u32(b, v) : v;
}

static nir_alu_type
store_type(const nir_def *v)
{
   return v->bit_size < 32 ? (nir_alu_type)(nir_type_uint | v->bit_size) : nir_type_invalid;
}

/* OpenCL private memory that stays memory (dynamically indexed arrays) comes
 * as scratch access: each invocation of the dispatch has a slot of
 * scratch_size bytes in the private buffer, addressed as global memory. */
static bool
lower_private_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned size = *(unsigned *)data;
   const bool load = intr->intrinsic == nir_intrinsic_load_scratch;

   if (!load && intr->intrinsic != nir_intrinsic_store_scratch)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *wid = nir_load_workgroup_id(b);
   nir_def *num = nir_load_num_workgroups(b);
   nir_def *group = nir_iadd(b, nir_channel(b, wid, 0),
                             nir_imul(b, nir_channel(b, num, 0),
                                      nir_iadd(b, nir_channel(b, wid, 1),
                                               nir_imul(b, nir_channel(b, num, 1),
                                                        nir_channel(b, wid, 2)))));
   nir_def *block;
   if (b->shader->info.workgroup_size_variable) {
      block = nir_load_workgroup_size(b);
   } else {
      const uint16_t *s = b->shader->info.workgroup_size;
      block = nir_imm_ivec3(b, s[0], s[1], s[2]);
   }
   nir_def *threads = nir_imul(b, nir_channel(b, block, 0),
                               nir_imul(b, nir_channel(b, block, 1), nir_channel(b, block, 2)));
   nir_def *slot = nir_iadd(b, nir_imul(b, group, threads), nir_load_local_invocation_index(b));
   nir_def *offset = intr->src[load ? 0 : 1].ssa;
   nir_def *addr = nir_iadd(b, nir_load_scratch_base_ptr(b, 1, 32, .base = 1),
                            nir_iadd(b, nir_imul_imm(b, slot, size), offset));
   /* slots are 16-byte aligned */
   const unsigned align_mul = MIN2(nir_intrinsic_align_mul(intr), 16);
   const unsigned align_offset = nir_intrinsic_align_offset(intr) % align_mul;

   if (load) {
      nir_def *repl = nir_load_global(b, intr->def.num_components, intr->def.bit_size, addr,
                                      .align_mul = align_mul, .align_offset = align_offset);
      nir_def_replace(&intr->def, repl);
   } else {
      nir_store_global(b, intr->src[0].ssa, addr, .write_mask = nir_intrinsic_write_mask(intr),
                       .align_mul = align_mul, .align_offset = align_offset);
      nir_instr_remove(&intr->instr);
   }

   return true;
}

/* Shared memory lives in DDR, one slice per workgroup: address = shared base +
 * linear workgroup index * slice size + offset. barrier() keeps the threads of
 * a group in step, and the slices of different groups never overlap. */
static bool
lower_shared_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const unsigned slice = *(unsigned *)data;
   unsigned off_src;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      off_src = 0;
      break;
   case nir_intrinsic_store_shared:
      off_src = 1;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *wid = nir_load_workgroup_id(b);
   nir_def *num = nir_load_num_workgroups(b);
   nir_def *group = nir_iadd(b, nir_channel(b, wid, 0),
                             nir_imul(b, nir_channel(b, num, 0),
                                      nir_iadd(b, nir_channel(b, wid, 1),
                                               nir_imul(b, nir_channel(b, num, 1),
                                                        nir_channel(b, wid, 2)))));
   /* with __local kernel arguments the slice size is the dispatch's */
   nir_def *slice_size = b->shader->info.cs.has_variable_shared_mem ?
                         nir_load_shared_slice_etna(b) : nir_imm_int(b, slice);
   nir_def *offset = nir_iadd(b, nir_imul(b, group, slice_size),
                              nir_iadd_imm(b, intr->src[off_src].ssa, nir_intrinsic_base(intr)));
   nir_def *addr = nir_load_shared_base_ptr(b, 1, 32);
   nir_def *repl = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
      repl = nir_load_global_etna(b, intr->def.num_components, intr->def.bit_size, addr, offset);
      break;
   case nir_intrinsic_store_shared:
      nir_store_global_etna(b, store_value32(b, intr->src[0].ssa), addr, offset,
                            .write_mask = nir_intrinsic_write_mask(intr),
                            .src_type = store_type(intr->src[0].ssa));
      break;
   case nir_intrinsic_shared_atomic:
      repl = nir_global_atomic_etna(b, intr->def.bit_size, addr, offset, intr->src[1].ssa,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   case nir_intrinsic_shared_atomic_swap:
      repl = nir_global_atomic_swap_etna(b, intr->def.bit_size, addr, offset,
                                         nir_vec2(b, intr->src[2].ssa, intr->src[1].ssa),
                                         .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   default:
      UNREACHABLE("not a shared memory access");
   }

   if (repl)
      nir_def_rewrite_uses(&intr->def, repl);
   nir_instr_remove(&intr->instr);
   return true;
}

enum etna_image_param {
   ETNA_IMAGE_ADDR,
   ETNA_IMAGE_STRIDE,
   ETNA_IMAGE_LAYER_STRIDE,
   ETNA_IMAGE_WIDTH,
   ETNA_IMAGE_HEIGHT,
   ETNA_IMAGE_DEPTH,
   ETNA_IMAGE_TILED,
   ETNA_IMAGE_PLANE_OFFSET,
};

static nir_def *
image_param(nir_builder *b, nir_def *index, enum etna_image_param param)
{
   return nir_load_image_param_etna(b, index, .base = param);
}

static void
image_type_size_align_1(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   unsigned s = glsl_type_is_array(type) ? glsl_get_aoa_size(type) : 1;

   *size = s;
   *align = s;
}

/* An OpenCL 2D texture read with img_load (key tex_img_read): fetches at
 * level 0, and, with a nearest, unnormalized sampler clamped to the edge or
 * to the border (key tex_img_sampled), samples at level 0, whose texel is the
 * floor of the coordinates clamped to the texture; outside it, the border:
 * (0, 0, 0, 0), or (0, 0, 0, 1) for a format without alpha. Other reads keep
 * the texture unit. */
static bool
lower_tex2d_to_image(nir_builder *b, nir_tex_instr *tex, const struct etna_shader_key *key)
{
   const bool sampled = key->tex_img_sampled & BITFIELD_BIT(tex->texture_index);
   const bool border = key->tex_img_border & BITFIELD_BIT(tex->texture_index);

   if ((tex->op != nir_texop_txf &&
        !(sampled && (tex->op == nir_texop_tex || tex->op == nir_texop_txl))) ||
       tex->is_shadow || tex->is_sparse || tex->def.bit_size != 32)
      return false;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_coord:
         break;
      case nir_tex_src_lod:
         /* 0 or 0.0 */
         if (!nir_src_is_const(tex->src[i].src) || nir_src_as_uint(tex->src[i].src) != 0)
            return false;
         break;
      default:
         return false;
      }
   }

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *index = nir_imm_int(b, 8 + tex->texture_index);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *xy = nir_trim_vector(b, nir_get_tex_src(tex, nir_tex_src_coord), 2);
   nir_def *inside = NULL;

   if (tex->op != nir_texop_txf) {
      /* the floor of each coordinate; integer coordinates converted to
       * float (read_imagef() with int2) are taken as they are */
      nir_def *texel[2];
      for (unsigned c = 0; c < 2; c++) {
         nir_scalar f = nir_scalar_resolved(xy, c);

         if (nir_scalar_is_alu(f) && nir_scalar_alu_op(f) == nir_op_i2f32 &&
             nir_scalar_chase_alu_src(f, 0).def->bit_size == 32)
            texel[c] = nir_mov_scalar(b, nir_scalar_chase_alu_src(f, 0));
         else
            texel[c] = nir_f2i32(b, nir_ffloor(b, nir_channel(b, xy, c)));
      }

      nir_def *size = nir_image_size(b, 2, 32, index, zero, .image_dim = GLSL_SAMPLER_DIM_2D,
                                     .image_array = false);
      nir_def *texel_xy = nir_vec2(b, texel[0], texel[1]);
      xy = nir_imin(b, nir_imax(b, texel_xy, nir_imm_ivec2(b, 0, 0)), nir_iadd_imm(b, size, -1));
      /* the texel is in the texture where clamping doesn't move it */
      if (border)
         inside = nir_ball_iequal(b, xy, texel_xy);
   }

   nir_def *repl = nir_image_load(b, tex->def.num_components, 32, index,
                                  nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1), zero, zero),
                                  zero, zero, .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
                                  .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_WRITEABLE,
                                  .dest_type = tex->dest_type);

   if (inside) {
      const bool a1 = key->tex_img_border_a1 & BITFIELD_BIT(tex->texture_index);
      nir_def *one = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float
                        ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
      nir_def *comps[4] = { zero, zero, zero, a1 ? one : zero };

      repl = nir_bcsel(b, inside, repl, nir_vec(b, comps, tex->def.num_components));
   }

   nir_def_replace(&tex->def, repl);
   return true;
}

/* OpenCL 1D image buffers that are read only come as buffer textures, which
 * the texture unit can't sample, and OpenCL 2D textures are read
 * faster by the image instructions (lower_tex2d_to_image()): read them with
 * those, as images 8 + texture index (see etna_uniform_image()). */
static bool
lower_texbuf_to_image(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;
   /* textures 0-7, or all in a wide dispatch (images 8-39) */
   const unsigned max = key->cs_tex_wide ? 32 : 8;

   if (tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array && tex->texture_index < max &&
       (key->tex_img_read & BITFIELD_BIT(tex->texture_index)))
      return lower_tex2d_to_image(b, tex, key);

   if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || tex->texture_index >= max ||
       (tex->op != nir_texop_txf && tex->op != nir_texop_txs))
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *index = nir_imm_int(b, 8 + tex->texture_index);
   nir_def *repl;

   if (tex->op == nir_texop_txs) {
      repl = nir_image_size(b, tex->def.num_components, 32, index, nir_imm_int(b, 0),
                            .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false);
   } else {
      nir_def *x = nir_channel(b, nir_get_tex_src(tex, nir_tex_src_coord), 0);
      nir_def *zero = nir_imm_int(b, 0);

      repl = nir_image_load(b, tex->def.num_components, 32, index,
                            nir_vec4(b, x, zero, zero, zero), zero, zero,
                            .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false,
                            .format = PIPE_FORMAT_NONE, .access = ACCESS_NON_WRITEABLE,
                            .dest_type = tex->dest_type);
   }

   nir_def_replace(&tex->def, repl);
   return true;
}

/* A wide compute dispatch (key cs_tex_wide: OpenCL, more than 16 textures):
 * the textures the texture unit still reads after lower_texbuf_to_image() take
 * the fragment slots in order of their index, where etna_launch_grid() binds
 * them. The passes after this one see slots, as for 16 textures or fewer. A
 * variant that needs more than 16 slots isn't run (etna_compute_probe_wide());
 * its extra textures share the last slot so that it compiles. */
static void
remap_tex_slots(nir_shader *s, struct etna_shader_variant *v)
{
   uint32_t used = 0;

   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               used |= BITFIELD_BIT(nir_instr_as_tex(instr)->texture_index);
         }
      }
   }

   memset(v->cs_tex_slot, 0xff, sizeof(v->cs_tex_slot));
   v->cs_tex_slots = 0;
   u_foreach_bit(t, used)
      v->cs_tex_slot[t] = v->cs_tex_slots++;

   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const unsigned slot = MIN2(v->cs_tex_slot[tex->texture_index], 15);

            tex->texture_index = slot;
            tex->sampler_index = slot;
         }
      }
   }
}

/* Is image index (0-7 the shader images, 8-39 the textures read as images)
 * 128-bit, or one component? Without a format in the shader (OpenCL) the key
 * says. */
static bool
key_img_bit(const struct etna_shader_key *key, nir_src index, bool comp1)
{
   if (!nir_src_is_const(index))
      return false;

   const unsigned i = nir_src_as_uint(index);

   if (i < 16)
      return (comp1 ? key->img_1comp : key->img_128bit) & BITFIELD_BIT(i);

   return i - 8 < 32 && ((comp1 ? key->tex_img_1comp : key->tex_img_128bit) & BITFIELD_BIT(i - 8));
}

/* Images are linear or 4x4 tiled (see etna_resource_create()). Loads and
 * stores use img_load/img_store, which address and convert texels through
 * the image descriptor. 128-bit formats are two stacked planes of 64-bit
 * texels (xy, then zw), see etna_resource_level_second_plane_offset(),
 * except in buffers, which hold them as 16-byte texels.
 * Atomics (r32ui, r32i) address the texel in the shader and use the atomic
 * instructions: linear base + layer * layer stride + y * stride + x * 4,
 * tiled in tiles of 4x4 texels in rows of stride * 4 bytes. */
static bool
lower_image_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct etna_shader_key *key = data;

   b->cursor = nir_before_instr(&intr->instr);

   /* image derefs to binding indices, like gl_nir_lower_images() */
   if (intr->intrinsic == nir_intrinsic_image_deref_load ||
       intr->intrinsic == nir_intrinsic_image_deref_store ||
       intr->intrinsic == nir_intrinsic_image_deref_size ||
       intr->intrinsic == nir_intrinsic_image_deref_atomic ||
       intr->intrinsic == nir_intrinsic_image_deref_atomic_swap) {
      nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
      nir_variable *var = nir_deref_instr_get_variable(deref);
      nir_def *index = nir_iadd_imm(b, nir_build_deref_offset(b, deref, image_type_size_align_1),
                                    var->data.driver_location);

      nir_rewrite_image_intrinsic(intr, index, nir_image_intrinsic_type_default);
   }

   if (intr->intrinsic != nir_intrinsic_image_load &&
       intr->intrinsic != nir_intrinsic_image_store &&
       intr->intrinsic != nir_intrinsic_image_size &&
       intr->intrinsic != nir_intrinsic_image_atomic &&
       intr->intrinsic != nir_intrinsic_image_atomic_swap)
      return false;

   nir_def *index = intr->src[0].ssa;
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);
   const bool layered = array || dim == GLSL_SAMPLER_DIM_3D || dim == GLSL_SAMPLER_DIM_CUBE;
   /* 1D and buffer images have one row: y is 0, a 1D array has its layer in y */
   const bool one_row = dim == GLSL_SAMPLER_DIM_1D || dim == GLSL_SAMPLER_DIM_BUF;

   if (intr->intrinsic == nir_intrinsic_image_size) {
      nir_def *size[3] = {
         image_param(b, index, ETNA_IMAGE_WIDTH),
         image_param(b, index, ETNA_IMAGE_HEIGHT),
         image_param(b, index, ETNA_IMAGE_DEPTH),
      };
      if (dim == GLSL_SAMPLER_DIM_CUBE && array)
         size[2] = nir_udiv_imm(b, size[2], 6);
      /* a 1D array has its layers second */
      if (dim == GLSL_SAMPLER_DIM_1D && array)
         size[1] = size[2];
      nir_def_replace(&intr->def, nir_vec(b, size, intr->def.num_components));
      return true;
   }

   const enum pipe_format format = nir_intrinsic_format(intr);
   const unsigned bpp = util_format_get_blocksize(format);
   /* without a format (OpenCL), the key says which images are 128-bit */
   const bool is_128bit = bpp == 16 ||
                          (format == PIPE_FORMAT_NONE && key_img_bit(key, intr->src[0], false));
   const bool planes = is_128bit && dim != GLSL_SAMPLER_DIM_BUF;
   const unsigned texel_bytes = planes ? 8 : bpp;
   nir_def *coord = intr->src[1].ssa;

   /* Without a format in the shader (OpenCL) the descriptor converts, but the
    * instruction still takes the register type and the components from the
    * format (emit_image()): give it a stand-in of the access's type, with one
    * component, four, or a 128-bit format (two per plane). */
   enum pipe_format emit_format = format;
   if (format == PIPE_FORMAT_NONE &&
       (intr->intrinsic == nir_intrinsic_image_load || intr->intrinsic == nir_intrinsic_image_store)) {
      const bool load = intr->intrinsic == nir_intrinsic_image_load;
      const nir_alu_type base = nir_alu_type_get_base_type(load ? nir_intrinsic_dest_type(intr)
                                                                : nir_intrinsic_src_type(intr));
      const bool one = key_img_bit(key, intr->src[0], true);
      static const enum pipe_format stand_in[3][3] = {
         /* float, uint, int */
         { PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_UINT, PIPE_FORMAT_R32_SINT },
         { PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R16G16B16A16_SINT },
         { PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R32G32B32A32_SINT },
      };
      const unsigned t = base == nir_type_uint ? 1 : base == nir_type_int ? 2 : 0;

      emit_format = stand_in[is_128bit ? 2 : one ? 0 : 1][t];
   }

   if (one_row) {
      nir_def *zero = nir_imm_int(b, 0);
      coord = nir_vec4(b, nir_channel(b, coord, 0), zero,
                       array ? nir_channel(b, coord, 1) : zero, zero);
   }

   /* img_load/img_store with the image descriptor. The 3D form takes the
    * address of the layer as its z coordinate. 128-bit formats are two
    * planes of 64-bit texels, xy then zw, each with its own descriptor. */
   if (intr->intrinsic == nir_intrinsic_image_load || intr->intrinsic == nir_intrinsic_image_store) {
      nir_def *hcoord[2];

      for (unsigned p = 0; p < 1 + planes; p++) {
         if (layered) {
            nir_def *base = image_param(b, index, ETNA_IMAGE_ADDR);
            if (p)
               base = nir_iadd(b, base, image_param(b, index, ETNA_IMAGE_PLANE_OFFSET));
            nir_def *layer = nir_iadd(b, base, nir_imul(b, nir_channel(b, coord, 2),
                                                        image_param(b, index, ETNA_IMAGE_LAYER_STRIDE)));
            hcoord[p] = nir_vec3(b, nir_channel(b, coord, 0), nir_channel(b, coord, 1), layer);
         } else {
            hcoord[p] = nir_trim_vector(b, coord, 2);
         }
      }

      if (intr->intrinsic == nir_intrinsic_image_load) {
         nir_def *t = nir_load_image_etna(b, index, hcoord[0], .component = 0,
                                          .image_dim = dim, .image_array = array, .format = emit_format);
         if (planes) {
            nir_def *hi = nir_load_image_etna(b, index, hcoord[1], .component = 1,
                                              .image_dim = dim, .image_array = array, .format = emit_format);
            t = nir_vec4(b, nir_channel(b, t, 0), nir_channel(b, t, 1),
                         nir_channel(b, hi, 0), nir_channel(b, hi, 1));
         } else if (format != PIPE_FORMAT_NONE ? util_format_get_nr_components(format) == 1 :
                    key_img_bit(key, intr->src[0], true)) {
            /* the load writes .x only: (r, 0, 0, 1); without a format in the
             * shader (OpenCL) the key says which images, and the result type
             * the type of the 1 */
            const bool int_one = format != PIPE_FORMAT_NONE ? util_format_is_pure_integer(format) :
               nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) != nir_type_float;
            nir_def *one = int_one ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0);
            t = nir_vec4(b, nir_channel(b, t, 0), nir_imm_int(b, 0), nir_imm_int(b, 0), one);
         }
         nir_def_replace(&intr->def, nir_trim_vector(b, t, intr->def.num_components));
      } else {
         nir_def *v = intr->src[3].ssa;
         nir_store_image_etna(b, index, hcoord[0], v, .component = 0,
                              .image_dim = dim, .image_array = array, .format = emit_format);
         if (planes)
            nir_store_image_etna(b, index, hcoord[1], nir_swizzle(b, v, (unsigned[]){ 2, 3, 2, 3 }, 4),
                                 .component = 1, .image_dim = dim,
                                 .image_array = array, .format = emit_format);
         nir_instr_remove(&intr->instr);
      }
      return true;
   }

   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = dim == GLSL_SAMPLER_DIM_1D ? nir_imm_int(b, 0) : nir_channel(b, coord, 1);
   nir_def *stride = image_param(b, index, ETNA_IMAGE_STRIDE);

   nir_def *linear = nir_iadd(b, nir_imul_imm(b, x, texel_bytes), nir_imul(b, y, stride));
   nir_def *in_tile = nir_iadd(b, nir_ishl_imm(b, nir_iand_imm(b, y, 3), 2), nir_iand_imm(b, x, 3));
   nir_def *tiled =
      nir_iadd(b, nir_imul(b, nir_ushr_imm(b, y, 2), nir_ishl_imm(b, stride, 2)),
               nir_imul_imm(b, nir_iadd(b, nir_ishl_imm(b, nir_ushr_imm(b, x, 2), 4), in_tile),
                            texel_bytes));
   nir_def *offset = nir_bcsel(b, nir_ine_imm(b, image_param(b, index, ETNA_IMAGE_TILED), 0),
                               tiled, linear);

   if (layered)
      offset = nir_iadd(b, offset, nir_imul(b, nir_channel(b, coord, 2),
                                            image_param(b, index, ETNA_IMAGE_LAYER_STRIDE)));

   nir_def *addr = image_param(b, index, ETNA_IMAGE_ADDR);

   /* r32ui and r32i only */
   if (intr->intrinsic == nir_intrinsic_image_atomic) {
      nir_def_replace(&intr->def,
                      nir_global_atomic_etna(b, 32, addr, offset, intr->src[3].ssa,
                                             .atomic_op = nir_intrinsic_atomic_op(intr)));
   } else {
      nir_def_replace(&intr->def,
                      nir_global_atomic_swap_etna(b, 32, addr, offset,
                                                  nir_vec2(b, intr->src[4].ssa, intr->src[3].ssa),
                                                  .atomic_op = nir_intrinsic_atomic_op(intr)));
   }

   return true;
}

/* an ALU instruction with a 64-bit source or result */
static bool
is_64bit_alu(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size == 64)
      return true;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      if (nir_src_bit_size(alu->src[i].src) == 64)
         return true;
   }
   return false;
}

/* Memory is accessed in aligned 32-bit words: narrower or unaligned loads
 * become word loads plus extraction, narrower stores atomic and/or on the
 * word (OpenCL char and short data, 8-bit kernel arguments). */
static nir_mem_access_size_align
etna_mem_access_size_align(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size,
                           uint32_t align_mul, uint32_t align_offset, bool offset_is_const,
                           enum gl_access_qualifier access, const void *cb_data)
{
   /* STORE.u8 / STORE.u16 write one narrow value: no atomics for a store
    * that isn't whole 32-bit words */
   const uint32_t align = nir_combined_align(align_mul, align_offset);

   /* LOAD.u8 / LOAD.u16 read one narrow value, zero-extended: no word
    * load and extraction for one that isn't whole 32-bit words */
   if ((intrin == nir_intrinsic_load_global || intrin == nir_intrinsic_load_global_constant) &&
       bit_size < 32 && align >= bit_size / 8 && (align < 4 || bytes % 4)) {
      return (nir_mem_access_size_align){
         .num_components = 1,
         .bit_size = bit_size,
         .align = bit_size / 8,
         .shift = nir_mem_access_shift_method_scalar,
      };
   }
   if ((intrin == nir_intrinsic_store_global || intrin == nir_intrinsic_store_shared) &&
       bit_size < 32 && align >= bit_size / 8 && (align < 4 || bytes % 4)) {
      return (nir_mem_access_size_align){
         .num_components = 1,
         .bit_size = bit_size,
         .align = bit_size / 8,
         .shift = nir_mem_access_shift_method_scalar,
      };
   }

   return (nir_mem_access_size_align){
      .num_components = MIN2(DIV_ROUND_UP(bytes, 4), 4),
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

/* adjacent 32-bit accesses become one access of up to four components: LOAD
 * and STORE take any 4-byte aligned address */
static bool
etna_vectorize_mem_cb(unsigned align_mul, unsigned align_offset, unsigned bit_size,
                      unsigned num_components, int64_t hole_size,
                      nir_intrinsic_instr *low, nir_intrinsic_instr *high, void *data)
{
   if (hole_size > 0 || bit_size != 32 || num_components > 4)
      return false;
   return nir_combined_align(align_mul, align_offset) >= 4;
}

/* The sequences below do half precision in fp32, for every input exactly as
 * the conversions would: fp32 ops round to nearest even and flush fp32
 * denormals on inputs and results (half denormals are normal fp32 values),
 * MAD is unfused (every product here is exact, so fusing wouldn't change a
 * result), integer ops work on the bits. They are vector code: no shift left
 * by a constant (lower_bitops makes it a scalar IMULLO0), BIT_INSERT instead.
 * The hardware CONV to and from f16 would replace them where it is exact. */

/* The 32-bit value whose low 16 bits are the 16-bit scalar s, the upper
 * bits unknown, and the shift that brings them down: through the widening
 * and narrowing that nir_lower_bit_size() leaves around a vector of 16-bit
 * values, to the 32-bit value or the half of a 32-bit word that it comes
 * from. */
static bool
low16_source(nir_scalar s, nir_scalar *word, unsigned *shift)
{
   s = nir_scalar_chase_movs(s);
   while (nir_scalar_is_alu(s)) {
      const nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_unpack_32_2x16 || op == nir_op_unpack_32_2x16_split_x ||
          op == nir_op_unpack_32_2x16_split_y) {
         *shift = op == nir_op_unpack_32_2x16_split_y || (op == nir_op_unpack_32_2x16 && s.comp) ? 16 : 0;
         *word = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
         return true;
      }
      if (op != nir_op_u2u16 && op != nir_op_i2i16)
         return false;

      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
      if (s.def->bit_size != 32)
         return false;
      /* u2u16(u2u32(x)) of a 16-bit x: x again */
      if (!nir_scalar_is_alu(s) ||
          (nir_scalar_alu_op(s) != nir_op_u2u32 && nir_scalar_alu_op(s) != nir_op_i2i32) ||
          nir_scalar_chase_alu_src(s, 0).def->bit_size != 16) {
         *shift = 0;
         *word = s;
         return true;
      }
      s = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   }
   return false;
}

/* 32-bit values whose low 16 bits are the components of a 16-bit value v,
 * the upper bits unknown (low16_source(), or u2u32 of the component), for
 * code that reads only the low 16 bits and then needs no mask. */
static nir_def *
low16_bits(nir_builder *b, nir_def *v)
{
   nir_scalar words[NIR_MAX_VEC_COMPONENTS];
   nir_const_value shifts[NIR_MAX_VEC_COMPONENTS];
   bool shifted = false;

   for (unsigned i = 0; i < v->num_components; i++) {
      unsigned shift = 0;
      if (!low16_source(nir_get_scalar(v, i), &words[i], &shift))
         words[i] = nir_get_scalar(nir_u2u32(b, nir_channel(b, v, i)), 0);
      shifts[i] = nir_const_value_for_uint(shift, 32);
      shifted |= shift;
   }

   nir_def *w = nir_vec_scalars(b, words, v->num_components);
   return shifted ? nir_ushr(b, w, nir_build_imm(b, v->num_components, 32, shifts)) : w;
}

/* pack_32_2x16 of two 16-bit values (a 16-bit vector store, from
 * nir_lower_mem_access_bit_sizes()): one BIT_INSERT of the words they come
 * from, where the lowered pack masks both and shifts the high one with a
 * scalar IMULLO0 (lower_bitops). halti5, as the compute shaders that have it. */
static bool
lower_pack_32_2x16(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_pack_32_2x16)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *w = low16_bits(b, nir_ssa_for_alu_src(b, alu, 0));
   nir_def_replace(&alu->def, nir_bitfield_insert(b, nir_channel(b, w, 0), nir_channel(b, w, 1),
                                                  nir_imm_int(b, 16), nir_imm_int(b, 16)));
   return true;
}

/* A 16-bit float as a 32-bit one, exactly. 0x70000000 | (h & 0x7fff) << 13
 * rebiases the half's exponent by 224: infinity and NaN come out as fp32
 * infinity and NaN with the payload, a finite half as its value times 2^112.
 * f = that * 2^-112 is a normal half's value; for a denormal half (exponent
 * 0) it is 2^-15 + m * 2^-25, and g = 2f - 2^-14 is m * 2^-24 exactly, while
 * g >= f for a normal half. The smaller of f, g and the rebiased bits (all
 * non-negative: an unsigned min of their bits) is the value, and for infinity
 * and NaN the rebiased bits themselves. */
static nir_def *
half_to_float(nir_builder *b, nir_def *half)
{
   nir_def *h = low16_bits(b, half);
   nir_def *biased = nir_bitfield_insert(b, nir_imm_int(b, 0x70000000), h,
                                         nir_imm_int(b, 13), nir_imm_int(b, 15));

   const unsigned save_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl |= nir_fp_exact;
   nir_def *f = nir_fmul_imm(b, biased, 0x1p-112);
   nir_def *g = nir_fmad(b, biased, nir_imm_float(b, 0x1p-111f), nir_imm_float(b, -0x1p-14f));
   b->fp_math_ctrl = save_math_ctrl;

   nir_def *abs = nir_umin(b, nir_umin(b, f, g), biased);
   return nir_bitfield_insert(b, abs, nir_ushr_imm(b, h, 15), nir_imm_int(b, 31), nir_imm_int(b, 1));
}

/* An fp32 value rounded to the nearest half, kept in fp32: f2f32(f2f16(x))
 * without the conversions. With E the exponent of x clamped to [-14, 15],
 * 2^(E+13) + |x| rounds |x| to a multiple of 2^(E-10), the half's ulp
 * (2^-24 below 2^-14, where halves are denormal), to nearest even: 2^(E+13)
 * is an even multiple of it. Subtracting 2^(E+13) again is exact. A result of
 * 65536 or more (|x| from 65520 up) times 2^112 overflows to infinity, and
 * the scale back is exact for every half. The sign comes from x, also for a
 * zero. Infinity stays infinity, NaN stays NaN. */
static nir_def *
round_to_half(nir_builder *b, nir_def *x)
{
   nir_def *pow2 = nir_umin_imm(b, nir_umax_imm(b, nir_iand_imm(b, x, 0x7f800000), 0x38800000),
                                0x47000000);

   const unsigned save_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl |= nir_fp_exact;
   nir_def *t = nir_fmad(b, pow2, nir_imm_float(b, 8192.0f), nir_fabs(b, x));
   nir_def *r = nir_fmad(b, pow2, nir_imm_float(b, -8192.0f), t);
   r = nir_fmul_imm(b, nir_fmul_imm(b, r, 0x1p112), 0x1p-112);
   b->fp_math_ctrl = save_math_ctrl;

   return nir_bitfield_insert(b, x, r, nir_imm_int(b, 0), nir_imm_int(b, 31));
}

/* An fp32 value to half bits, rounded to nearest even (f2f16, f2f16_rtne),
 * in the low 16 bits. |x| clamped to 65536 (the top of the rounding to half,
 * from 65520 on), 2^(E+13) + |x| rounds as in round_to_half(): its bits are
 * those of 2^(E+13) plus k, |x| in units of the half's ulp, 0 to 2048. k plus
 * (E + 14) << 10 is the half (k = 2048 carries into the exponent, 65536 into
 * infinity); 2^(E+13) has no bits in the low 16, (E + 14) << 10 is
 * 2^E's bits >> 13, plus 0x3c00 in the low 16 bits. A NaN gives 0x7e00, as
 * nir_lower_fp16_casts(). */
static nir_def *
float_to_half_rtne(nir_builder *b, nir_def *x)
{
   nir_def *abs = nir_iand_imm(b, x, 0x7fffffff);
   nir_def *a = nir_umin_imm(b, abs, 0x47800000);
   nir_def *pow2 = nir_umax_imm(b, nir_iand_imm(b, a, 0x7f800000), 0x38800000);

   const unsigned save_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl |= nir_fp_exact;
   nir_def *t = nir_fmad(b, pow2, nir_imm_float(b, 8192.0f), a);
   b->fp_math_ctrl = save_math_ctrl;

   nir_def *h = nir_iadd(b, t, nir_iadd_imm(b, nir_ushr_imm(b, pow2, 13), 0x3c00));
   h = nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000), abs), nir_imm_int(b, 0x7e00), h);
   return nir_bitfield_insert(b, nir_ushr_imm(b, x, 16), h, nir_imm_int(b, 0), nir_imm_int(b, 15));
}

/* Whether a 32-bit float is the value of a half (or NaN) without rounding: a
 * half converted, a constant that is one, a value rounded to half before
 * (in rounded), or what moves, neg, abs, sat, min, max and selects make of
 * those: each returns one of its operands, negated, or 0 or 1. */
static bool
is_half_value(nir_scalar s, struct set *rounded, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      const float v = uif(nir_scalar_as_uint(s));
      return isnan(v) || fui(_mesa_half_to_float(_mesa_float_to_half(v))) == fui(v);
   }

   if (!nir_scalar_is_alu(s) || !depth)
      return false;
   if (_mesa_set_search(rounded, s.def))
      return true;

   unsigned first = 0;
   switch (nir_scalar_alu_op(s)) {
   case nir_op_f2f32:
      return nir_scalar_chase_alu_src(s, 0).def->bit_size == 16;
   case nir_op_bcsel:
      first = 1;
      FALLTHROUGH;
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fsat:
   case nir_op_fmin:
   case nir_op_fmax:
      for (unsigned i = first; i < nir_op_infos[nir_scalar_alu_op(s)].num_inputs; i++) {
         if (!is_half_value(nir_scalar_chase_alu_src(s, i), rounded, depth - 1))
            return false;
      }
      return true;
   default:
      return false;
   }
}

/* f2f32(f2f16(x)) of an fp32 x, which nir_lower_bit_size() leaves between two
 * widened half operations: round x to half precision without leaving fp32,
 * all components at once, or keep x where it is a half already. The f2f16
 * stays for other uses (a store), or goes as dead code. */
static bool
fold_half_round_trip(nir_builder *b, nir_alu_instr *alu, void *data)
{
   struct set *rounded = data;

   if (alu->op != nir_op_f2f32 || alu->src[0].src.ssa->bit_size != 16)
      return false;

   if (!nir_def_is_alu(alu->src[0].src.ssa))
      return false;

   nir_alu_instr *narrow = nir_instr_as_alu(nir_def_instr(alu->src[0].src.ssa));
   if ((narrow->op != nir_op_f2f16 && narrow->op != nir_op_f2f16_rtne) ||
       narrow->src[0].src.ssa->bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   unsigned swizzle[NIR_MAX_VEC_COMPONENTS];
   bool exact = true;
   for (unsigned i = 0; i < alu->def.num_components; i++) {
      swizzle[i] = narrow->src[0].swizzle[alu->src[0].swizzle[i]];
      exact &= is_half_value(nir_get_scalar(narrow->src[0].src.ssa, swizzle[i]), rounded, 8);
   }

   nir_def *x = nir_swizzle(b, narrow->src[0].src.ssa, swizzle, alu->def.num_components);
   nir_def *res = exact ? x : round_to_half(b, x);
   _mesa_set_add(rounded, res);
   nir_def_replace(&alu->def, res);
   return true;
}

static bool
in_loop(nir_instr *instr, nir_loop *loop)
{
   for (nir_cf_node *n = &instr->block->cf_node; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

/* A half carried by a phi, a loop's accumulator for example, is after
 * nir_lower_bit_size() a 32-bit phi of its bits, which the arithmetic
 * converts to fp32 at every use, and every iteration converts its new value
 * to half bits. Carry the fp32 value in a phi of its own instead: its sources
 * are f2f32 of the half sources, and f2f32(f2f16(x)) folds into
 * round_to_half(x). Uses of the bits after the loop keep the bits phi, but
 * its sources that are a conversion to half become 0x10000 (no 16-bit
 * value): there the bits are f2f16 of the fp32 value, which is what the
 * conversion gave (f2f16 of x rounded to half is f2f16(x), also for a NaN);
 * a source that wasn't converted (an initial value) keeps its bits. */
static bool
carry_half_phi_as_float(nir_builder *b, nir_phi_instr *phi)
{
   if (phi->def.bit_size != 32)
      return false;

   nir_loop *loop = NULL;
   nir_cf_node *parent = phi->instr.block->cf_node.parent;
   if (parent->type == nir_cf_node_loop &&
       nir_loop_first_block(nir_cf_node_as_loop(parent)) == phi->instr.block)
      loop = nir_cf_node_as_loop(parent);

   /* the uses: u2u16 of the phi (nir_lower_bit_size()), used by f2f32 or,
    * after the loop, by anything but a phi */
   nir_alu_instr *narrow[16];
   unsigned num_narrow = 0;
   bool any_float = false, any_bits = false;
   nir_foreach_use_including_if(src, &phi->def) {
      if (nir_src_is_if(src) || num_narrow == ARRAY_SIZE(narrow))
         return false;
      nir_instr *use = nir_src_use_instr(src);
      if (use->type != nir_instr_type_alu || nir_instr_as_alu(use)->op != nir_op_u2u16)
         return false;
      narrow[num_narrow++] = nir_instr_as_alu(use);
      nir_foreach_use_including_if(usrc, &nir_instr_as_alu(use)->def) {
         if (nir_src_is_if(usrc))
            return false;
         nir_instr *user = nir_src_use_instr(usrc);
         if (user->type == nir_instr_type_alu && nir_instr_as_alu(user)->op == nir_op_f2f32) {
            any_float = true;
            continue;
         }
         if (!loop || user->type == nir_instr_type_phi || in_loop(user, loop))
            return false;
         any_bits = true;
      }
   }
   if (!any_float)
      return false;

   /* the sources: u2u32 of a half, or a constant one (nir_lower_bit_size()
    * folds u2u32 of a constant) */
   nir_foreach_phi_src(src, phi) {
      if (nir_src_is_const(src->src)) {
         for (unsigned i = 0; i < phi->def.num_components; i++) {
            if (nir_src_comp_as_uint(src->src, i) > 0xffff)
               return false;
         }
      } else if (!nir_def_is_alu(src->src.ssa) || nir_def_as_alu(src->src.ssa)->op != nir_op_u2u32 ||
                 nir_def_as_alu(src->src.ssa)->src[0].src.ssa->bit_size != 16) {
         return false;
      }
   }

   nir_const_value marker[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < phi->def.num_components; i++)
      marker[i] = nir_const_value_for_uint(0x10000, 32);

   nir_phi_instr *fphi = nir_phi_instr_create(b->shader);
   nir_def_init(&fphi->instr, &fphi->def, phi->def.num_components, 32);
   bool marked = false;
   nir_foreach_phi_src(src, phi) {
      b->cursor = nir_after_block_before_jump(src->pred);
      nir_def *half = nir_src_is_const(src->src) ? nir_u2u16(b, src->src.ssa) :
                      nir_ssa_for_alu_src(b, nir_def_as_alu(src->src.ssa), 0);
      nir_phi_instr_add_src(fphi, src->pred, nir_f2f32(b, half));
      if (any_bits && nir_def_is_alu(half) &&
          (nir_def_as_alu(half)->op == nir_op_f2f16 || nir_def_as_alu(half)->op == nir_op_f2f16_rtne)) {
         nir_src_rewrite(&src->src, nir_build_imm(b, phi->def.num_components, 32, marker));
         marked = true;
      }
   }
   nir_instr_insert_after(&phi->instr, &fphi->instr);

   for (unsigned n = 0; n < num_narrow; n++) {
      nir_alu_instr *u = narrow[n];
      nir_foreach_use_safe(usrc, &u->def) {
         nir_instr *user = nir_src_use_instr(usrc);
         b->cursor = nir_before_instr(user);
         if (user->type == nir_instr_type_alu && nir_instr_as_alu(user)->op == nir_op_f2f32) {
            nir_alu_instr *cvt = nir_instr_as_alu(user);
            unsigned swizzle[NIR_MAX_VEC_COMPONENTS];
            for (unsigned i = 0; i < cvt->def.num_components; i++)
               swizzle[i] = u->src[0].swizzle[cvt->src[0].swizzle[i]];
            nir_def_replace(&cvt->def, nir_swizzle(b, &fphi->def, swizzle, cvt->def.num_components));
         } else if (marked) {
            nir_def *bits = nir_bcsel(b, nir_ieq_imm(b, &phi->def, 0x10000),
                                      nir_u2u32(b, nir_f2f16(b, &fphi->def)), &phi->def);
            unsigned swizzle[NIR_MAX_VEC_COMPONENTS];
            for (unsigned i = 0; i < u->def.num_components; i++)
               swizzle[i] = u->src[0].swizzle[i];
            nir_src_rewrite(usrc, nir_u2u16(b, nir_swizzle(b, bits, swizzle, u->def.num_components)));
         }
      }
   }
   return true;
}

static bool
carry_half_phis_as_float(nir_shader *s)
{
   bool progress = false;
   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_phi_safe(phi, block)
            impl_progress |= carry_half_phi_as_float(&b, phi);
      }
      nir_progress(impl_progress, impl, nir_metadata_control_flow);
      progress |= impl_progress;
   }
   return progress;
}

static bool
fold_half_round_trips(nir_shader *s)
{
   struct set *rounded = _mesa_pointer_set_create(NULL);
   bool progress = nir_shader_alu_pass(s, fold_half_round_trip, nir_metadata_control_flow, rounded);
   _mesa_set_destroy(rounded, NULL);
   return progress;
}

/* vecN(f2f32(a), f2f32(b), ...) of 16-bit scalars (vload_halfN with 2-byte
 * aligned loads): f2f32(vecN(a, b, ...)), one vector conversion */
static bool
gather_half_conversions(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (!nir_op_is_vec(alu->op) || alu->def.bit_size != 32)
      return false;

   nir_scalar halves[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; i++) {
      nir_def *src = alu->src[i].src.ssa;
      if (!nir_def_is_alu(src) || nir_def_as_alu(src)->op != nir_op_f2f32 ||
          nir_def_as_alu(src)->src[0].src.ssa->bit_size != 16 || !list_is_singular(&src->uses))
         return false;
      halves[i] = nir_scalar_chase_alu_src(nir_get_scalar(src, alu->src[i].swizzle[0]), 0);
   }

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, nir_f2f32(b, nir_vec_scalars(b, halves, alu->def.num_components)));
   return true;
}

/* No 16-bit float ALU: conversions from a half go through its exact 32-bit
 * value, those to a half from integers through a 32-bit float, and those
 * rounding to nearest even from a 32-bit float through float_to_half_rtne();
 * the other rounding modes stay for nir_lower_fp16_casts(). */
static bool
lower_fp16_conversion(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (!nir_op_infos[alu->op].is_conversion)
      return false;

   const nir_alu_type src_type = nir_op_infos[alu->op].input_types[0];
   const nir_alu_type dst_type = nir_op_infos[alu->op].output_type;
   const unsigned src_bits = alu->src[0].src.ssa->bit_size;
   const bool from_half = nir_alu_type_get_base_type(src_type) == nir_type_float && src_bits == 16;
   const bool to_half = nir_alu_type_get_base_type(dst_type) == nir_type_float && alu->def.bit_size == 16;
   const bool rtne = alu->op == nir_op_f2f16_rtne ||
                     (alu->op == nir_op_f2f16 &&
                      !(b->shader->info.float_controls_execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16));
   const bool from_float32 = nir_alu_type_get_base_type(src_type) == nir_type_float && src_bits == 32;

   if (!from_half && !(to_half && nir_alu_type_get_base_type(src_type) != nir_type_float) &&
       !(to_half && from_float32 && rtne))
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *src = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *res;

   if (from_half && to_half) {
      res = src;
   } else if (from_half) {
      res = nir_type_convert(b, half_to_float(b, src), nir_type_float32,
                             nir_alu_type_get_base_type(dst_type) | alu->def.bit_size,
                             nir_rounding_mode_undef);
   } else {
      nir_def *f = from_float32 ? src :
                   nir_type_convert(b, src, nir_alu_type_get_base_type(src_type) | src_bits,
                                    nir_type_float32, nir_rounding_mode_undef);
      res = nir_u2u16(b, float_to_half_rtne(b, f));
   }

   nir_def_replace(&alu->def, res);
   return true;
}

/* OpenCL read_imageh and write_imageh: the texture unit and the image
 * instructions work on 32-bit values. A 16-bit result is the 32-bit one
 * converted (to nearest even: a texel of a half format converts exactly, one
 * of an 8-bit normalized format rounds), a 16-bit value to store is widened
 * first (exactly). */
static bool
widen_16bit_image_access(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   nir_def *def;
   nir_alu_type type;

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (tex->def.bit_size != 16)
         return false;
      def = &tex->def;
      type = nir_alu_type_get_base_type(tex->dest_type);
      tex->dest_type = type | 32;
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_bindless_image_load:
         if (intr->def.bit_size != 16)
            return false;
         def = &intr->def;
         type = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));
         nir_intrinsic_set_dest_type(intr, type | 32);
         break;
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_bindless_image_store: {
         nir_def *value = intr->src[3].ssa;
         if (value->bit_size != 16)
            return false;
         type = nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
         b->cursor = nir_before_instr(instr);
         nir_src_rewrite(&intr->src[3], nir_type_convert(b, value, type | 16, type | 32,
                                                         nir_rounding_mode_undef));
         nir_intrinsic_set_src_type(intr, type | 32);
         return true;
      }
      default:
         return false;
      }
   } else {
      return false;
   }

   def->bit_size = 32;
   b->cursor = nir_after_instr(instr);
   nir_def *narrow = nir_type_convert(b, def, type | 32, type | 16, nir_rounding_mode_undef);
   nir_def_rewrite_uses_after(def, narrow);
   return true;
}

/* no 8- or 16-bit arithmetic: do it in 32 bits */
static unsigned
etna_lower_bit_size_cb(const nir_instr *instr, UNUSED void *data)
{
   if (instr->type == nir_instr_type_phi) {
      const unsigned size = nir_instr_as_phi(instr)->def.bit_size;
      return size == 8 || size == 16 ? 32 : 0;
   }

   if (instr->type != nir_instr_type_alu)
      return 0;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   unsigned size = alu->def.bit_size;

   /* conversions have sized types: fold_narrow_conversion() handles them */
   if (nir_op_infos[alu->op].is_conversion)
      return 0;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
      if (alu->src[i].src.ssa->bit_size != 1)
         size = MIN2(size == 1 ? 32 : size, alu->src[i].src.ssa->bit_size);

   return size == 8 || size == 16 ? 32 : 0;
}

/* After nir_lower_bit_size(): widen(narrow(x)) of a 32-bit x is a mask or a
 * sign extension in 32 bits (with lower_extract_byte, nir_opt_algebraic
 * leaves u2u32(u2u8(x)) alone, and the 8-bit value would reach the backend) */
static bool
fold_narrow_conversion(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   bool sign, to_float;

   /* f2u8(x) of a 32-bit float is u2u8(f2u32(x)), b2i8(x) is u2u8(b2i32(x)),
    * u2u16(x) of an 8-bit x is u2u16(u2u32(x)): split them, so that only the
    * narrowing of a 32-bit integer is left, folded below into its widening
    * user */
   if (alu->def.bit_size == 8 || alu->def.bit_size == 16) {
      const unsigned src_size = alu->src[0].src.ssa->bit_size;
      nir_def *wide = NULL;

      b->cursor = nir_before_instr(&alu->instr);
      switch (alu->op) {
      case nir_op_f2u8:
      case nir_op_f2u16:
         if (src_size == 32)
            wide = nir_f2u32(b, nir_ssa_for_alu_src(b, alu, 0));
         break;
      case nir_op_f2i8:
      case nir_op_f2i16:
         if (src_size == 32)
            wide = nir_f2i32(b, nir_ssa_for_alu_src(b, alu, 0));
         break;
      case nir_op_b2i8:
      case nir_op_b2i16:
         wide = nir_b2i32(b, nir_ssa_for_alu_src(b, alu, 0));
         break;
      case nir_op_u2u8:
      case nir_op_u2u16:
         if (src_size == 8 || src_size == 16)
            wide = nir_u2u32(b, nir_ssa_for_alu_src(b, alu, 0));
         break;
      case nir_op_i2i8:
      case nir_op_i2i16:
         if (src_size == 8 || src_size == 16)
            wide = nir_i2i32(b, nir_ssa_for_alu_src(b, alu, 0));
         break;
      default:
         break;
      }

      if (wide) {
         nir_def_replace(&alu->def, nir_u2uN(b, wide, alu->def.bit_size));
         return true;
      }
   }

   switch (alu->op) {
   case nir_op_u2u32: sign = false; to_float = false; break;
   case nir_op_i2i32: sign = true; to_float = false; break;
   case nir_op_u2f32: sign = false; to_float = true; break;
   case nir_op_i2f32: sign = true; to_float = true; break;
   default:
      return false;
   }

   if (!nir_def_is_alu(alu->src[0].src.ssa))
      return false;

   nir_alu_instr *narrow = nir_instr_as_alu(nir_def_instr(alu->src[0].src.ssa));
   if ((narrow->op != nir_op_u2u8 && narrow->op != nir_op_u2u16 &&
        narrow->op != nir_op_i2i8 && narrow->op != nir_op_i2i16) ||
       narrow->src[0].src.ssa->bit_size != 32)
      return false;

   const unsigned bits = narrow->def.bit_size;
   b->cursor = nir_before_instr(&alu->instr);

   /* a value LOAD.u8/u16 zero-extended is its own unsigned widening */
   nir_def *src32 = narrow->src[0].src.ssa;
   if (!sign && !to_float && nir_def_is_intrinsic(src32)) {
      nir_intrinsic_instr *ld = nir_def_as_intrinsic(src32);
      if (ld->intrinsic == nir_intrinsic_load_global_etna &&
          nir_alu_type_get_type_size(nir_intrinsic_dest_type(ld)) == bits &&
          (narrow->op == nir_op_u2u8 || narrow->op == nir_op_u2u16)) {
         nir_def_replace(&alu->def, src32);
         return true;
      }
   }

   unsigned swiz[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; i++)
      swiz[i] = alu->src[0].swizzle[i];
   nir_def *x = nir_swizzle(b, nir_ssa_for_alu_src(b, narrow, 0), swiz, alu->def.num_components);
   nir_def *v = sign ? nir_ishr_imm(b, nir_ishl_imm(b, x, 32 - bits), 32 - bits) :
                       nir_iand_imm(b, x, BITFIELD_MASK(bits));
   if (to_float)
      v = sign ? nir_i2f32(b, v) : nir_u2f32(b, v);

   nir_def_replace(&alu->def, v);
   return true;
}

/* The low 8 or 16 bits of a sum, product or bitwise op depend only on
 * the low 8 or 16 bits of its sources, and a product by 2^k (a left shift by
 * k) needs k bits less of the other source than its users need of it: the
 * sign extension of an 8-bit value is ishr(imul(x, 1 << 24), 24).
 * narrow_use_bits() is the number of low bits of a value that one use needs,
 * from need[], the bits the users of each def need (0: none known yet).
 */
static unsigned
narrow_class(unsigned bits)
{
   return bits <= 8 ? 8 : bits <= 16 ? 16 : 32;
}

static unsigned
narrow_shift(nir_alu_instr *alu, unsigned other, bool shift)
{
   if (!nir_src_is_const(alu->src[other].src))
      return 0;

   const uint32_t v = nir_src_comp_as_uint(alu->src[other].src, alu->src[other].swizzle[0]);
   for (unsigned i = 1; i < alu->def.num_components; i++) {
      if (nir_src_comp_as_uint(alu->src[other].src, alu->src[other].swizzle[i]) != v)
         return 0;
   }

   if (shift)
      return v & 31;
   return util_is_power_of_two_nonzero(v) ? ffs(v) - 1 : 0;
}

static unsigned
narrow_use_bits(nir_src *src, const uint8_t *need)
{
   if (nir_src_is_if(src))
      return 32;

   nir_instr *instr = nir_src_use_instr(src);
   if (instr->type == nir_instr_type_phi)
      return need[nir_instr_as_phi(instr)->def.index];

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      /* the value of a store that the emitter makes 8 or 16 bits wide */
      if (intr->intrinsic == nir_intrinsic_store_global_etna && src == &intr->src[0]) {
         const nir_alu_type type = nir_intrinsic_src_type(intr);
         return type == nir_type_uint8 ? 8 : type == nir_type_uint16 ? 16 : 32;
      }
      return 32;
   }

   if (instr->type != nir_instr_type_alu)
      return 32;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size != 32 && !nir_op_infos[alu->op].is_conversion)
      return 32;

   const unsigned idx = container_of(src, nir_alu_src, src) - alu->src;
   const unsigned bits = need[alu->def.index];

   switch (alu->op) {
   case nir_op_u2u8:
   case nir_op_i2i8:
      return 8;
   case nir_op_u2u16:
   case nir_op_i2i16:
      return 16;
   case nir_op_iand: {
      /* a mask */
      const unsigned other = 1 - idx;
      if (nir_src_is_const(alu->src[other].src)) {
         uint32_t mask = 0;
         for (unsigned i = 0; i < alu->def.num_components; i++)
            mask |= nir_src_comp_as_uint(alu->src[other].src, alu->src[other].swizzle[i]);
         return MIN2(bits, mask <= 0xff ? 8 : mask <= 0xffff ? 16 : 32);
      }
      return bits;
   }
   case nir_op_imul: {
      const unsigned k = narrow_shift(alu, 1 - idx, false);
      return narrow_class(bits > k ? bits - k : 1);
   }
   case nir_op_ishl: {
      if (idx != 0)
         return 32;
      const unsigned k = narrow_shift(alu, 1, true);
      return narrow_class(bits > k ? bits - k : 1);
   }
   case nir_op_iadd:
   case nir_op_ineg:
   case nir_op_imad:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_inot:
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
      return bits;
   default:
      return 32;
   }
}

/* OpenCL char and short arithmetic is lowered to 32 bits. A product or
 * multiply-add whose users need only its low 8 or 16 bits becomes a lane
 * product (imul8x4_etna and the others): one IMULLO0 or IMADLO0 for every
 * component of a vector, where a 32-bit one takes one per component.
 */
static bool
etna_narrow_products(nir_shader *s)
{
   bool progress = false;

   nir_foreach_function_impl(impl, s) {
      bool impl_progress = false;
      nir_index_ssa_defs(impl);
      uint8_t *need = calloc(impl->ssa_alloc, 1);
      if (!need)
         return progress;

      /* users before defs, until the phis of loops settle: need[] only grows */
      bool changed;
      do {
         changed = false;
         nir_foreach_block_reverse(block, impl) {
            nir_foreach_instr_reverse(instr, block) {
               nir_def *def = nir_instr_def(instr);
               if (!def)
                  continue;

               unsigned bits = 0;
               nir_foreach_use_including_if(src, def) {
                  bits = MAX2(bits, narrow_use_bits(src, need));
                  if (bits == 32)
                     break;
               }
               if (bits > need[def->index]) {
                  need[def->index] = bits;
                  changed = true;
               }
            }
         }
      } while (changed);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            const unsigned bits = need[alu->def.index];
            if (alu->def.bit_size != 32 || !bits || bits == 32)
               continue;

            if (alu->op == nir_op_imul)
               alu->op = bits == 8 ? nir_op_imul8x4_etna : nir_op_imul16x2_etna;
            else if (alu->op == nir_op_imad)
               alu->op = bits == 8 ? nir_op_imad8x4_etna : nir_op_imad16x2_etna;
            else
               continue;
            /* the bits above the lane can wrap */
            alu->no_signed_wrap = false;
            alu->no_unsigned_wrap = false;
            impl_progress = true;
         }
      }

      free(need);
      nir_progress(impl_progress, impl, nir_metadata_control_flow);
      progress |= impl_progress;
   }

   return progress;
}

/* the two operands of a global access for an OpenCL address: LOAD and STORE
 * add them, so base + offset needs no add of its own (and no register for an
 * offset of 0) */
static void
split_global_address(nir_builder *b, nir_def *addr, nir_def **base, nir_def **offset)
{
   nir_instr *instr = nir_def_instr(addr);
   nir_alu_instr *alu = instr->type == nir_instr_type_alu ? nir_instr_as_alu(instr) : NULL;

   if (alu && alu->op == nir_op_iadd && alu->def.num_components == 1) {
      const bool const0 = nir_src_is_const(alu->src[0].src);
      *base = nir_ssa_for_alu_src(b, alu, const0 ? 1 : 0);
      *offset = nir_ssa_for_alu_src(b, alu, const0 ? 0 : 1);
      return;
   }

   *base = addr;
   *offset = nir_imm_int(b, 0);
}

/* LOAD and STORE shift their second address operand left by up to 7 bits
 * (LEFT_SHIFT) before they add it to the first, so an offset that is an index
 * scaled by a power of two needs no shift or multiply of its own: the index
 * goes in as the offset, the scale as OFFSET_SHIFT. One instruction less for
 * every indexed access (NXP's compiler emits load.ls2, store.ls6, ...). */
static bool
fold_offset_shift(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   unsigned off;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_etna:
      off = 1;
      break;
   case nir_intrinsic_store_global_etna:
      off = 2;
      break;
   default:
      return false;
   }

   if (nir_intrinsic_offset_shift(intr))
      return false;

   nir_scalar s = nir_scalar_resolved(intr->src[off].ssa, 0);
   if (!nir_scalar_is_alu(s))
      return false;

   nir_op op = nir_scalar_alu_op(s);
   nir_scalar src0 = nir_scalar_chase_alu_src(s, 0);
   nir_scalar src1 = nir_scalar_chase_alu_src(s, 1);
   nir_scalar index;
   uint64_t shift;

   if (op == nir_op_ishl && nir_scalar_is_const(src1)) {
      index = src0;
      shift = nir_scalar_as_uint(src1) & 31;
   } else if (op == nir_op_imul && (nir_scalar_is_const(src0) || nir_scalar_is_const(src1))) {
      const bool c0 = nir_scalar_is_const(src0);
      const uint64_t scale = nir_scalar_as_uint(c0 ? src0 : src1);
      if (!util_is_power_of_two_nonzero64(scale))
         return false;
      index = c0 ? src1 : src0;
      shift = util_logbase2_64(scale);
   } else {
      return false;
   }

   if (shift == 0 || shift > 7)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[off], nir_channel(b, index.def, index.comp));
   nir_intrinsic_set_offset_shift(intr, shift);
   return true;
}

/* STORE.u8/u16 writes the low byte or half of its value: a mask of the
 * value that keeps those bits (the widening of an 8- or 16-bit value) is
 * not needed */
static bool
drop_narrow_store_mask(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_global_etna)
      return false;

   const unsigned bits = nir_alu_type_get_type_size(nir_intrinsic_src_type(intr));
   if ((bits != 8 && bits != 16) || intr->src[0].ssa->num_components != 1)
      return false;

   nir_scalar v = nir_scalar_resolved(intr->src[0].ssa, 0);
   if (!nir_scalar_is_alu(v) || nir_scalar_alu_op(v) != nir_op_iand)
      return false;

   for (unsigned i = 0; i < 2; i++) {
      nir_scalar m = nir_scalar_chase_alu_src(v, i);
      if (!nir_scalar_is_const(m))
         continue;
      const uint32_t keep = BITFIELD_MASK(bits);
      if ((nir_scalar_as_uint(m) & keep) != keep)
         continue;
      nir_scalar x = nir_scalar_chase_alu_src(v, 1 - i);
      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[0], nir_channel(b, x.def, x.comp));
      return true;
   }
   return false;
}

/* SSBO access becomes address + offset: the address is a uniform the driver
 * fills with the GPU address of the bound buffer. OpenCL global access
 * (32-bit addresses) is the same, the address split in two. */
static bool
lower_ssbo_access(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_def *base, *offset;

   nir_def *repl = NULL;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      split_global_address(b, intr->src[0].ssa, &base, &offset);
      if (intr->def.bit_size == 8 || intr->def.bit_size == 16) {
         /* LOAD.u8/u16 zero-extends into the 32-bit register */
         assert(intr->def.num_components == 1);
         repl = nir_u2uN(b, nir_load_global_etna(b, 1, 32, base, offset,
                                                 .dest_type = intr->def.bit_size == 8 ? nir_type_uint8
                                                                                      : nir_type_uint16),
                         intr->def.bit_size);
      } else {
         repl = nir_load_global_etna(b, intr->def.num_components, intr->def.bit_size, base, offset);
      }
      break;
   case nir_intrinsic_store_global:
      split_global_address(b, intr->src[1].ssa, &base, &offset);
      /* dual16: STORE.f16 stores each work-item's half */
      if (data && ((const struct etna_shader_variant *)data)->dual16 &&
          intr->src[0].ssa->bit_size == 16) {
         nir_store_global_etna(b, intr->src[0].ssa, base, offset,
                               .write_mask = nir_intrinsic_write_mask(intr),
                               .src_type = nir_type_float16);
         break;
      }
      nir_store_global_etna(b, store_value32(b, intr->src[0].ssa), base, offset,
                            .write_mask = nir_intrinsic_write_mask(intr),
                            .src_type = store_type(intr->src[0].ssa));
      break;
   case nir_intrinsic_global_atomic:
      repl = nir_global_atomic_etna(b, intr->def.bit_size, intr->src[0].ssa, nir_imm_int(b, 0),
                                    intr->src[1].ssa,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   case nir_intrinsic_global_atomic_swap:
      repl = nir_global_atomic_swap_etna(b, intr->def.bit_size, intr->src[0].ssa,
                                         nir_imm_int(b, 0),
                                         nir_vec2(b, intr->src[2].ssa, intr->src[1].ssa),
                                         .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   case nir_intrinsic_load_ssbo:
      repl = nir_load_global_etna(b, intr->def.num_components, intr->def.bit_size,
                                  nir_load_ssbo_address_etna(b, intr->src[0].ssa),
                                  intr->src[1].ssa);
      break;
   case nir_intrinsic_store_ssbo:
      nir_store_global_etna(b, intr->src[0].ssa,
                            nir_load_ssbo_address_etna(b, intr->src[1].ssa),
                            intr->src[2].ssa,
                            .write_mask = nir_intrinsic_write_mask(intr));
      break;
   case nir_intrinsic_ssbo_atomic:
      repl = nir_global_atomic_etna(b, intr->def.bit_size,
                                    nir_load_ssbo_address_etna(b, intr->src[0].ssa),
                                    intr->src[1].ssa, intr->src[2].ssa,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   case nir_intrinsic_ssbo_atomic_swap:
      repl = nir_global_atomic_swap_etna(b, intr->def.bit_size,
                                         nir_load_ssbo_address_etna(b, intr->src[0].ssa),
                                         intr->src[1].ssa,
                                         nir_vec2(b, intr->src[3].ssa, intr->src[2].ssa),
                                         .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   default:
      return false;
   }

   if (repl)
      nir_def_rewrite_uses(&intr->def, repl);
   nir_instr_remove(&intr->instr);
   return true;
}

static uint8_t
alu_width_cb(const nir_instr *instr, UNUSED const void *cb_data)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_ball_fequal2:
      case nir_op_ball_fequal3:
      case nir_op_ball_fequal4:
      case nir_op_bany_fnequal2:
      case nir_op_bany_fnequal3:
      case nir_op_bany_fnequal4:
      case nir_op_ball_iequal2:
      case nir_op_ball_iequal3:
      case nir_op_ball_iequal4:
      case nir_op_bany_inequal2:
      case nir_op_bany_inequal3:
      case nir_op_bany_inequal4: return 1;
      case nir_op_fdot2:
      case nir_op_fdot3:
      case nir_op_fdot4: return 0;
      default: break;
      }
   }

   return 4;
}

/* An intrinsic takes a scalar operand as a whole value, so a vector op that
 * only computes addresses (loads and stores of single components) needs a mov
 * per component back out of the vector: keep those scalar */
static uint8_t
etna_vectorize_cb(const nir_instr *instr, UNUSED const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 4;

   const nir_def *def = &nir_instr_as_alu((nir_instr *)instr)->def;
   bool any = false;

   nir_foreach_use_including_if(src, def) {
      if (nir_src_is_if(src) || nir_src_use_instr(src)->type != nir_instr_type_intrinsic)
         return 4;
      any = true;
   }
   return any ? 1 : 4;
}

/* registers have four components: a wider phi (an OpenCL float8 carried
 * around a loop that isn't unrolled) is split into phis of four */
static uint8_t
phi_width_cb(UNUSED const nir_instr *instr, UNUSED const void *data)
{
   return 4;
}

/* the copies that keep the segments of etna_nir_fold_workgroup() apart from
 * CSE, once no CSE runs any more */
static bool
lower_opaque_copy(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_opaque_copy_etna)
      return false;

   nir_def_replace(&intr->def, intr->src[0].ssa);
   return true;
}

/* nir_lower_frexp() emits fisfinite, which only nir_opt_algebraic_late
 * lowers, after the bool lowering has turned it into fisfinite32. Test the
 * exponent bits: a float compare against infinity doesn't work here. */
static bool
lower_fisfinite(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_fisfinite)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   uint64_t exp_mask = x->bit_size == 16 ? 0x7c00 : 0x7f800000;
   nir_def *mask = nir_imm_intN_t(b, exp_mask, x->bit_size);
   nir_def_replace(&alu->def, nir_ine(b, nir_iand(b, x, mask), mask));
   return true;
}

/* SELECT tests its sources itself: SELECT.NZ x, a, b is x != 0 ? a : b, and
 * SELECT.EQ x, x, y is y where x is NaN. So b32csel(ine32(x, 0), a, b) needs
 * no CMP (ieq32 swaps a and b), and neither does b32csel(feq32(x, x), x, y)
 * (fneu32 swaps): fsel_ord_etna(x, y). */
static bool
fold_select_test(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if (alu->op != nir_op_b32csel)
      return false;

   nir_instr *ci = nir_def_instr(alu->src[0].src.ssa);
   if (ci->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cmp = nir_instr_as_alu(ci);
   const unsigned n = alu->def.num_components;

   if (cmp->op == nir_op_feq32 || cmp->op == nir_op_fneu32) {
      /* the value selected where x == x must be x itself */
      const unsigned keep = cmp->op == nir_op_feq32 ? 1 : 2;
      if (cmp->src[0].src.ssa != cmp->src[1].src.ssa ||
          cmp->src[0].src.ssa != alu->src[keep].src.ssa || alu->def.bit_size != 32)
         return false;
      for (unsigned i = 0; i < n; i++) {
         const unsigned c = alu->src[0].swizzle[i];
         if (cmp->src[0].swizzle[c] != alu->src[keep].swizzle[i] ||
             cmp->src[1].swizzle[c] != alu->src[keep].swizzle[i])
            return false;
      }
      b->cursor = nir_before_instr(&alu->instr);
      nir_def_replace(&alu->def, nir_fsel_ord_etna(b, nir_ssa_for_alu_src(b, alu, keep),
                                                   nir_ssa_for_alu_src(b, alu, 3 - keep)));
      return true;
   }

   if (cmp->op != nir_op_ine32 && cmp->op != nir_op_ieq32)
      return false;

   int zero = -1;
   for (unsigned s = 0; s < 2 && zero < 0; s++) {
      if (!nir_src_is_const(cmp->src[s].src))
         continue;
      bool all = true;
      for (unsigned i = 0; i < n; i++) {
         const unsigned c = cmp->src[s].swizzle[alu->src[0].swizzle[i]];
         all = all && nir_src_comp_as_uint(cmp->src[s].src, c) == 0;
      }
      if (all)
         zero = s;
   }
   if (zero < 0 || nir_src_bit_size(cmp->src[1 - zero].src) != 32)
      return false;

   unsigned swiz[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
      swiz[i] = alu->src[0].swizzle[i];

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = nir_swizzle(b, nir_ssa_for_alu_src(b, cmp, 1 - zero), swiz, n);
   nir_def *t = nir_ssa_for_alu_src(b, alu, 1), *f = nir_ssa_for_alu_src(b, alu, 2);
   nir_def_replace(&alu->def, cmp->op == nir_op_ine32 ? nir_b32csel(b, x, t, f) : nir_b32csel(b, x, f, t));
   return true;
}

/* gl_HelperInvocation: no hardware source is known on GC7000 (the internal
 * register group gives the front face in .x; no component or register
 * flags helper lanes). Report every invocation as a covered one: shaders
 * using it compile and run, but can't tell helper lanes apart. */
static bool
lower_helper_invocation(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_helper_invocation &&
       intr->intrinsic != nir_intrinsic_is_helper_invocation)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_false(b));
   return true;
}

/* Dual16 ----------------------------------------------------------------
 *
 * In dual16 mode (SH_CONFIG DUAL16) a hardware thread carries two work-items,
 * T0 and T1. A register component holds two halves, T0's in bits 0-15 and
 * T1's in bits 16-31, and an instruction without a thread select works on
 * both: half arithmetic at twice the rate. An instruction with a thread select
 * (.t0/.t1) runs for one work-item; it reads its half of a component (t) or
 * the full 32 bits (th), and writes its half (converted to fp16) or, with .hp,
 * the full 32 bits. The hardware puts id slot s of work-item t in register
 * 2s + t. Packed instructions round to nearest even and flush half denormals
 * (RMODE 0; RMODE 1 rounds toward zero). See cl/pk/dual16-notes.md.
 */

static bool
dual16_reject(const struct etna_compile *c, const char *why, const char *what)
{
   if (DBG_ENABLED(ETNA_DBG_COMPILER_MSGS)) {
      fprintf(stderr, "etnaviv: dual16: %s not taken: %s %s\n",
              nir_shader_get_entrypoint(c->nir)->function->name, why, what ? what : "");
   }
   return false;
}

/* 8- and 64-bit values are out; a half is a float, and only the operations
 * below handle one: arithmetic whose dual16 results were checked (mad is
 * OpenCL mad() or a contracted x * y + z: unfused, each rounded), moves, and
 * conversions to half rounding to nearest even. ffma (fma() must be fused),
 * division, sqrt, compares and the rest stay in single mode. */
static bool
dual16_alu_ok(const struct etna_compile *c, const nir_alu_instr *alu, bool *has_half)
{
   const char *name = nir_op_infos[alu->op].name;
   const unsigned n = nir_op_infos[alu->op].num_inputs;
   bool half = alu->def.bit_size == 16;

   if (alu->def.bit_size == 8 || alu->def.bit_size == 64)
      return dual16_reject(c, "8- or 64-bit alu", name);
   for (unsigned i = 0; i < n; i++) {
      const unsigned b = alu->src[i].src.ssa->bit_size;
      if (b == 8 || b == 64)
         return dual16_reject(c, "8- or 64-bit alu source", name);
      half |= b == 16;
   }
   if (!half)
      return true;

   *has_half = true;
   switch (alu->op) {
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_fmad:
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec5:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_fneg:
   case nir_op_fabs:
      for (unsigned i = 0; i < n; i++) {
         if (alu->src[i].src.ssa->bit_size != 16)
            return dual16_reject(c, "half op with a wider source", name);
      }
      return alu->def.bit_size == 16 || dual16_reject(c, "half op with a wider result", name);
   case nir_op_f2f16:
      /* the kernel's rounding of half conversions, which is round to nearest
       * even unless it asks for round toward zero */
      if (c->nir->info.float_controls_execution_mode & FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP16)
         return dual16_reject(c, "rounding mode", name);
      FALLTHROUGH;
   case nir_op_f2f16_rtne:
   case nir_op_u2f16:
   case nir_op_i2f16:
      return alu->src[0].src.ssa->bit_size == 32 || dual16_reject(c, "conversion", name);
   case nir_op_f2f32:
      return alu->src[0].src.ssa->bit_size == 16 || dual16_reject(c, "conversion", name);
   default:
      return dual16_reject(c, "half op", name);
   }
}

/* values that can differ between the two work-items of a thread: the ids
 * (the two can even be in different workgroups) and what is computed from
 * them; the rest comes from constants and arguments */
static bool
dual16_varying_intrinsic(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_workgroup_size:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_base_global_invocation_id:
   case nir_intrinsic_load_base_workgroup_id:
   case nir_intrinsic_load_work_dim:
      return false;
   default:
      return true;
   }
}

static bool
dual16_varying_src(nir_src *src, void *data)
{
   const BITSET_WORD *varying = data;
   return !BITSET_TEST(varying, src->ssa->index);
}

/* every branch condition is the same for both work-items of a thread: one
 * branch serves both */
static bool
dual16_uniform_cf(const struct etna_compile *c, nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   bool ok = true, progress;

   nir_index_ssa_defs(impl);
   BITSET_WORD *varying = rzalloc_array(NULL, BITSET_WORD, BITSET_WORDS(impl->ssa_alloc));
   if (!varying)
      return false;

   do {
      progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (!def || BITSET_TEST(varying, def->index))
               continue;

            bool v = false;
            switch (instr->type) {
            case nir_instr_type_intrinsic:
               v = dual16_varying_intrinsic(nir_instr_as_intrinsic(instr)) ||
                   !nir_foreach_src(instr, dual16_varying_src, varying);
               break;
            case nir_instr_type_alu:
            case nir_instr_type_phi:
               v = !nir_foreach_src(instr, dual16_varying_src, varying);
               break;
            default:
               break;
            }
            if (v) {
               BITSET_SET(varying, def->index);
               progress = true;
            }
         }
      }
   } while (progress);

   nir_foreach_block(block, impl) {
      nir_if *nif = nir_block_get_following_if(block);
      if (nif && BITSET_TEST(varying, nif->condition.ssa->index)) {
         ok = dual16_reject(c, "branch that differs between work-items", NULL);
         break;
      }
   }

   ralloc_free(varying);
   return ok;
}

/* the NIR rusticl hands over (late: the NIR to emit): an OpenCL kernel with
 * half arithmetic, no loads, atomics, barrier, shared or private memory, and
 * uniform control flow */
static bool
dual16_check(const struct etna_compile *c, nir_shader *s, bool late)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   bool has_half = false;

   if (s->info.stage != MESA_SHADER_COMPUTE || !s->info.cs.ptr_size ||
       DBG_ENABLED(ETNA_DBG_NO_DUAL16))
      return false;
   if (s->info.uses_control_barrier || s->info.shared_size || s->scratch_size ||
       s->info.cs.has_variable_shared_mem)
      return dual16_reject(c, "barrier, shared or private memory", NULL);
   if (!exec_list_is_singular(&impl->body)) {
      if (DBG_ENABLED(ETNA_DBG_NO_DUAL16_CF))
         return dual16_reject(c, "control flow", NULL);
      if (!dual16_uniform_cf(c, s))
         return false;
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_load_const:
         case nir_instr_type_undef:
         case nir_instr_type_phi:
            if (nir_instr_def(instr)->bit_size == 8 || nir_instr_def(instr)->bit_size == 64)
               return dual16_reject(c, "8- or 64-bit value", NULL);
            break;
         case nir_instr_type_jump:
            if (nir_instr_as_jump(instr)->type != nir_jump_break &&
                nir_instr_as_jump(instr)->type != nir_jump_continue)
               return dual16_reject(c, "jump", NULL);
            break;
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            /* wide vectors are split before emission (as in single mode) */
            if (late && nir_op_is_vec(alu->op) && alu->def.num_components > 4)
               return dual16_reject(c, "wide vector", NULL);
            if (!dual16_alu_ok(c, alu, &has_half))
               return false;
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const char *name = nir_intrinsic_infos[intr->intrinsic].name;
            switch (intr->intrinsic) {
            case nir_intrinsic_load_global_invocation_id:
            case nir_intrinsic_load_local_invocation_id:
            case nir_intrinsic_load_workgroup_id:
            case nir_intrinsic_load_local_invocation_index:
            case nir_intrinsic_load_base_global_invocation_id:
            case nir_intrinsic_load_kernel_input:
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_work_dim:
               if (late)
                  return dual16_reject(c, "intrinsic after lowering", name);
               break;
            case nir_intrinsic_load_input:
               if (!late)
                  return dual16_reject(c, "intrinsic", name);
               break;
            case nir_intrinsic_load_workgroup_size:
            case nir_intrinsic_load_num_workgroups:
            case nir_intrinsic_load_base_workgroup_id:
               break;
            case nir_intrinsic_load_uniform:
               if (late && !nir_src_is_const(intr->src[0]))
                  return dual16_reject(c, "indirect uniform", NULL);
               break;
            case nir_intrinsic_store_global:
            case nir_intrinsic_store_global_etna:
               if ((intr->intrinsic == nir_intrinsic_store_global_etna) != late)
                  return dual16_reject(c, "intrinsic", name);
               if (intr->src[0].ssa->bit_size == 8 || intr->src[0].ssa->bit_size == 64)
                  return dual16_reject(c, "8- or 64-bit store", NULL);
               break;
            default:
               return dual16_reject(c, "intrinsic", name);
            }
            if (nir_intrinsic_infos[intr->intrinsic].has_dest && intr->def.bit_size != 32)
               return dual16_reject(c, "not 32-bit", name);
            break;
         }
         default:
            return dual16_reject(c, "instruction type", NULL);
         }
      }
   }

   if (!late && !has_half)
      return dual16_reject(c, "no half arithmetic", NULL);
   return true;
}

static bool
dual16_early(const struct etna_compile *c, nir_shader *s)
{
   if (!dual16_check(c, s, false))
      return false;
   if (DBG_ENABLED(ETNA_DBG_COMPILER_MSGS))
      fprintf(stderr, "etnaviv: dual16: %s taken\n", nir_shader_get_entrypoint(s)->function->name);
   return true;
}

/* The single-mode code, with halves in 32-bit register components, becomes
 * dual16 code. Single-mode register r holds its packed halves in p0(r), its
 * 32-bit values in p0(r) for T0 and p1(r) for T1; the id registers
 * (r < cs_input_count) are the hardware's, slot r of T0 in 2r and of T1 in
 * 2r + 1 (CL_CONFIG value order 3: global id, local id, workgroup id; two
 * slots at least, as NXP's dual16 dispatches). A single-mode register
 * component never holds a half and a 32-bit value at the same time, so the
 * map is the same at every instruction. An instruction runs once (packed) if
 * its result and register sources are halves, else once per work-item:
 * 32-bit sources through th, a 32-bit result with .hp. A branch runs once on
 * T0's condition: a boolean (0 or ~0, alike in both halves) that both
 * work-items share (dual16_uniform_cf()). Branch targets move with the code. */
static bool
dual16_rewrite(struct etna_compile *c, struct etna_shader_variant *v, unsigned num_blocks)
{
   const unsigned ic = v->cs_input_count;
   const unsigned n = MAX2(v->num_temps, ic);
   const unsigned slots = MAX2(ic, 2);
   const unsigned base = 2 * slots;
   const unsigned total = base + 2 * (n - ic);
   const char *e = getenv("ETNA_DUAL16_RMODE");
   const unsigned rmode = e ? atoi(e) & 3 : 0;

   if (ic > 3 || total > 63)
      return dual16_reject(c, "registers", NULL);

#define P0(r) ((r) < ic ? 2 * (r) : base + (r) - ic)
#define P1(r) ((r) < ic ? 2 * (r) + 1 : base + (n - ic) + (r) - ic)
#define PT(r, t) ((t) ? P1(r) : P0(r))

   struct etna_inst *out = malloc(2 * (c->inst_ptr + 1) * sizeof(*out));
   unsigned *at = malloc((c->inst_ptr + 1) * sizeof(*at));
   unsigned m = 0;
   const char *why = NULL;

   if (!out || !at) {
      free(out);
      free(at);
      return false;
   }

   for (int i = 0; i < c->inst_ptr && !why; i++) {
      const struct etna_inst *in = &c->code[i];
      const struct etna_d16_inst *d = &c->d16[i];
      const bool packed_result = d->dst == ETNA_D16_P;
      bool any_w = false, p_on_dst = false;

      at[i] = m;

      switch (in->opcode) {
      case ISA_OPC_NOP:
      case ISA_OPC_BRANCH:
         out[m++] = *in;
         continue;
      case ISA_OPC_BRANCH_UNARY: {
         struct etna_inst o = *in;
         if (o.src[0].use && o.src[0].rgroup == ISA_REG_GROUP_TEMP) {
            if (o.src[0].amode || o.src[0].reg >= n) {
               why = "branch";
               break;
            }
            o.src[0].reg = P0(in->src[0].reg);
         }
         out[m++] = o;
         continue;
      }
      default:
         break;
      }
      if (why)
         break;

      if (!d->from_nir || in->dst.amode || (in->dst.use && in->dst.reg >= n)) {
         why = "instruction";
         break;
      }
      for (unsigned s = 0; s < ETNA_NUM_SRC; s++) {
         const struct etna_inst_src *src = &in->src[s];
         if (!src->use)
            continue;
         if (src->rgroup == ISA_REG_GROUP_IMMED && (src->imm_type == 0 || src->imm_type == 3)) {
            why = "float immediate";
            break;
         }
         if (src->rgroup != ISA_REG_GROUP_TEMP)
            continue;
         if (src->amode || src->reg >= n ||
             (d->src[s] != ETNA_D16_P && d->src[s] != ETNA_D16_W)) {
            why = "register source";
            break;
         }
         any_w |= d->src[s] == ETNA_D16_W;
         p_on_dst |= d->src[s] == ETNA_D16_P && in->dst.use && src->reg == in->dst.reg;
      }
      if (why)
         break;

      if (packed_result && !any_w && in->opcode != ISA_OPC_STORE) {
         struct etna_inst o = *in;
         if (o.dst.use)
            o.dst.reg = P0(in->dst.reg);
         for (unsigned s = 0; s < ETNA_NUM_SRC; s++) {
            if (o.src[s].use && o.src[s].rgroup == ISA_REG_GROUP_TEMP)
               o.src[s].reg = P0(in->src[s].reg);
         }
         if (!o.rounding)
            o.rounding = rmode;
         out[m++] = o;
         continue;
      }

      /* a 32-bit result of T0 overwrites both halves of its component: if a
       * packed source shares the register, T1 goes first */
      const bool t1_first = !packed_result && p_on_dst;
      for (unsigned k = 0; k < 2; k++) {
         const unsigned t = t1_first ? 1 - k : k;
         struct etna_inst o = *in;
         o.thread = t ? ISA_THREAD_T1 : ISA_THREAD_T0;
         if (o.dst.use && in->opcode != ISA_OPC_STORE) {
            o.dst.reg = packed_result ? P0(in->dst.reg) : PT(in->dst.reg, t);
            o.dst_full = !packed_result;
         }
         for (unsigned s = 0; s < ETNA_NUM_SRC; s++) {
            if (!o.src[s].use || o.src[s].rgroup != ISA_REG_GROUP_TEMP)
               continue;
            if (d->src[s] == ETNA_D16_W) {
               o.src[s].rgroup = ISA_REG_GROUP_TH;
               o.src[s].reg = PT(in->src[s].reg, t);
            } else {
               o.src[s].reg = P0(in->src[s].reg);
            }
         }
         out[m++] = o;
      }
   }

#undef PT
#undef P1
#undef P0

   if (!why && m > ETNA_MAX_SHADER_INSTRUCTIONS)
      why = "too long";
   if (why) {
      free(out);
      free(at);
      return dual16_reject(c, why, NULL);
   }

   at[c->inst_ptr] = m;
   for (unsigned b = 0; b < num_blocks; b++) {
      if (c->block_ptr[b] <= (unsigned)c->inst_ptr)
         c->block_ptr[b] = at[c->block_ptr[b]];
   }

   if (DBG_ENABLED(ETNA_DBG_COMPILER_MSGS))
      fprintf(stderr, "etnaviv: dual16: %s: %u instructions -> %u, %u temporaries -> %u, %u id slots\n",
              nir_shader_get_entrypoint(c->nir)->function->name, (unsigned)c->inst_ptr, m,
              v->num_temps, total, slots);

   free(at);
   free(c->code);
   c->code = out;
   c->code_alloc = 2 * (c->inst_ptr + 1);
   c->inst_ptr = m;
   v->num_temps = total;
   v->dual16_inputs = slots;
   return true;
}

static bool
compile_shader_one(struct etna_shader_variant *v, bool spill_constants, unsigned fold, bool fold_loop);

/* a dual16 compile that fails is retried in single mode */
static bool
compile_shader(struct etna_shader_variant *v, bool spill_constants, unsigned fold, bool fold_loop)
{
   if (compile_shader_one(v, spill_constants, fold, fold_loop))
      return true;
   if (!v || !v->dual16)
      return false;

   v->dual16_off = true;
   return compile_shader_one(v, spill_constants, fold, fold_loop);
}

static bool
compile_shader_one(struct etna_shader_variant *v, bool spill_constants, unsigned fold, bool fold_loop)
{
   if (unlikely(!v))
      return false;

   struct etna_compile *c = CALLOC_STRUCT(etna_compile);
   if (!c)
      return false;

   /* a retry starts from a clean variant */
   memset(VARIANT_CACHE_PTR(v), 0, VARIANT_CACHE_SIZE);
   v->cs_fold_loop = fold > 1 && fold_loop;

   c->variant = v;
   c->info = v->shader->info;
   c->specs = v->shader->specs;
   c->nir = nir_shader_clone(NULL, v->shader->nir);

   nir_shader *s = c->nir;
   const struct etna_specs *specs = c->specs;

   /* a variant for the workgroup size of the dispatch (key cs_block) */
   if (s->info.stage == MESA_SHADER_COMPUTE && v->key.cs_block[0]) {
      for (unsigned i = 0; i < 3; i++)
         s->info.workgroup_size[i] = v->key.cs_block[i];
      s->info.workgroup_size_variable = false;
   }

   v->stage = s->info.stage;
   v->uses_discard = s->info.fs.uses_discard;
   v->vs_id_in_reg = -1;
   v->vs_pos_out_reg = -1;
   v->vs_pointsize_out_reg = -1;
   v->ps_depth_out_reg = -1;

   if (s->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, s, nir_lower_fragcolor, v->shader->compiler->max_render_targets);

   /*
    * Lower glTexCoord, fixes e.g. neverball point sprite (exit cylinder stars)
    * and gl4es pointsprite.trace apitrace
    */
   if (s->info.stage == MESA_SHADER_FRAGMENT && v->key.sprite_coord_enable) {
      NIR_PASS(_, s, nir_lower_texcoord_replace, v->key.sprite_coord_enable,
                 false, v->key.sprite_coord_yinvert);
   }

   /*
    * Remove any dead in variables before we iterate over them
    */
   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_shader_in, NULL);

   /* lower_flrp32 only tells the frontend to run nir_lower_flrp(), which
    * rusticl doesn't (OpenCL mix()): the backend has no flrp */
   NIR_PASS(_, s, nir_lower_flrp, 16 | 32 | 64, false);

   /* rusticl lowers 64-bit integers before nir_lower_convert_alu_types(),
    * whose saturating and rounding conversions of OpenCL long make new 64-bit
    * compares, selects and conversions: lower those too. One component at a
    * time, so the unpacks of loaded values meet the packs of
    * nir_lower_mem_access_bit_sizes() per component, and cancel. */
   NIR_PASS(_, s, nir_lower_alu_to_scalar, is_64bit_alu, NULL);
   NIR_PASS(_, s, nir_lower_int64);

   if (s->info.stage == MESA_SHADER_COMPUTE) {
      v->uses_discard = false;
      v->cs_fold = 1;
      v->dual16 = fold <= 1 && !v->dual16_off && !v->key.cs_single && dual16_early(c, s);

      if (fold > 1) {
         if (!etna_nir_fold_workgroup(s, fold, fold_loop, &v->cs_scratch_slots)) {
            ralloc_free(c->nir);
            free(c->code);
            FREE(c);
            return false;
         }
         v->cs_fold = fold;

         bool progress;
         do {
            progress = false;
            NIR_PASS(progress, s, nir_opt_copy_prop);
            NIR_PASS(progress, s, nir_opt_constant_folding);
            NIR_PASS(progress, s, nir_opt_algebraic);
            NIR_PASS(progress, s, nir_opt_cse);
            NIR_PASS(progress, s, nir_opt_dce);
         } while (progress);
      }

      NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               glsl_get_cl_type_size_align);
      NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);

      const nir_lower_mem_access_bit_sizes_options mem_opts = {
         .callback = etna_mem_access_size_align,
         .modes = nir_var_mem_ubo | nir_var_mem_global | nir_var_mem_constant |
                  nir_var_mem_shared | nir_var_mem_ssbo,
         .may_lower_unaligned_stores_to_atomics = true,
      };
      v->cs_private_size = align(s->scratch_size, 16);
      if (v->cs_private_size)
         NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_private_access,
                  nir_metadata_control_flow, &v->cs_private_size);

      {
         const nir_load_store_vectorize_options lsv_opts = {
            .callback = etna_vectorize_mem_cb,
            .modes = nir_var_mem_global | nir_var_mem_shared,
         };
         bool vec = false;
         NIR_PASS(vec, s, nir_opt_load_store_vectorize, &lsv_opts);
         if (vec) {
            NIR_PASS(_, s, nir_opt_copy_prop);
            NIR_PASS(_, s, nir_opt_dce);
         }
      }

      bool narrow = false;
      NIR_PASS(narrow, s, nir_shader_instructions_pass, widen_16bit_image_access,
               nir_metadata_control_flow, NULL);
      NIR_PASS(narrow, s, nir_lower_mem_access_bit_sizes, &mem_opts);
      if (!v->dual16) { /* dual16: halves stay 16-bit */
         NIR_PASS(narrow, s, nir_lower_bit_size, etna_lower_bit_size_cb, NULL);
         NIR_PASS(narrow, s, carry_half_phis_as_float);
         NIR_PASS(narrow, s, fold_half_round_trips);
         NIR_PASS(narrow, s, nir_opt_dce);
         NIR_PASS(narrow, s, nir_shader_alu_pass, gather_half_conversions, nir_metadata_control_flow, NULL);
         NIR_PASS(narrow, s, nir_shader_alu_pass, lower_fp16_conversion, nir_metadata_none, NULL);
         NIR_PASS(narrow, s, nir_lower_fp16_casts, nir_lower_fp16_all);
         NIR_PASS(narrow, s, nir_shader_alu_pass, lower_pack_32_2x16, nir_metadata_control_flow, NULL);
      }
      NIR_PASS(narrow, s, nir_shader_alu_pass, fold_narrow_conversion,
               nir_metadata_control_flow, NULL);
      if (narrow) {
         NIR_PASS(_, s, nir_opt_constant_folding);
         NIR_PASS(_, s, nir_opt_algebraic);
         NIR_PASS(_, s, nir_opt_dce);
      }
      v->cs_shared_size = align(s->info.shared_size, 16);
      v->cs_uses_barrier = s->info.uses_control_barrier;
      if (v->cs_shared_size || s->info.cs.has_variable_shared_mem)
         NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_shared_access,
                  nir_metadata_none, &v->cs_shared_size);

      NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_cs_intrinsic,
               nir_metadata_control_flow, &v->cs_input_count);
   }

   NIR_PASS(_, s, nir_shader_tex_pass, lower_texbuf_to_image, nir_metadata_control_flow, &v->key);
   if (v->key.cs_tex_wide)
      remap_tex_slots(s, v);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_image_access,
            nir_metadata_control_flow, &v->key);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_ssbo_access,
            nir_metadata_control_flow, v);

   /* setup input linking */
   struct etna_shader_io_file *sf = &v->infile;
   if (s->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_in_variable(var, s) {
         unsigned idx = var->data.driver_location;
         sf->reg[idx].reg = idx;
         sf->reg[idx].slot = var->data.location;
         sf->reg[idx].interpolation = var->data.interpolation;
         sf->reg[idx].num_components = glsl_get_components(var->type);
         sf->num_reg = MAX2(sf->num_reg, idx+1);
      }
   } else {
      unsigned count = 0;
      nir_foreach_shader_in_variable(var, s) {
         unsigned idx = var->data.driver_location;
         sf->reg[idx].reg = idx + 1;
         sf->reg[idx].slot = var->data.location;
         if (var->data.interpolation == INTERP_MODE_NONE && v->key.flatshade &&
             (var->data.location == VARYING_SLOT_COL0 ||
              var->data.location == VARYING_SLOT_COL1)) {
            var->data.interpolation = INTERP_MODE_FLAT;
         }
         sf->reg[idx].interpolation = var->data.interpolation;
         sf->reg[idx].num_components = glsl_get_components(var->type);
         sf->num_reg = MAX2(sf->num_reg, idx+1);
         count++;
      }
      assert(sf->num_reg == count);
   }

   NIR_PASS(_, s, nir_lower_io, nir_var_shader_in | nir_var_uniform, etna_glsl_type_size,
            (nir_lower_io_options)0);

   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_lower_indirect_derefs_to_if_else_trees, nir_var_all,
            UINT32_MAX);

   if (v->key.use_xfb_emu)
      NIR_PASS(_, s, etna_nir_lower_xfb);

   NIR_PASS(_, s, etna_nir_lower_texture, &v->key, v->shader->info);
   NIR_PASS(_, s, etna_nir_lower_128bit, &v->key);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_helper_invocation,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, s, nir_lower_frexp);
   NIR_PASS(_, s, nir_shader_alu_pass, lower_fisfinite, nir_metadata_control_flow, NULL);
   /* nir_lower_int64() leaves phis 64-bit, joined to their 32-bit halves by
    * packs (OpenCL: libclc's software fma() in a loop): split them, and the
    * optimizations cancel the packs */
   NIR_PASS(_, s, nir_lower_64bit_phis);
   /* its unpacks of 64-bit vectors (OpenCL long3 lanes merged at an if) one
    * component at a time too */
   NIR_PASS(_, s, nir_lower_alu_to_scalar, is_64bit_alu, NULL);
   NIR_PASS(_, s, nir_lower_alu_width, alu_width_cb, NULL);
   NIR_PASS(_, s, nir_lower_phis_to_scalar, phi_width_cb, NULL);

   NIR_PASS(_, s, nir_lower_alu_to_scalar, etna_alu_to_scalar_filter_cb, c->info);
   if (c->info->halti >= 2) {
      nir_lower_idiv_options idiv_options = {
         .allow_fp16 = true,
      };
      /* by a constant: a multiply-high and shifts, before the general case */
      NIR_PASS(_, s, nir_opt_idiv_const, 32);
      NIR_PASS(_, s, nir_lower_idiv, &idiv_options);
   }
   NIR_PASS(_, s, nir_lower_alu);
   if (!s->options->no_integers)
      NIR_PASS(_, s, nir_shader_alu_pass, lower_fminmax_nan, nir_metadata_control_flow, NULL);

   etna_optimize_loop(s);

   /* TODO: remove this extra run if nir_opt_peephole_select is able to handle ubo's. */
   if (OPT(s, etna_nir_lower_ubo_to_uniform))
      etna_optimize_loop(s);

   NIR_PASS(_, s, etna_lower_io, v);
   NIR_PASS(_, s, nir_lower_pack);
   /* lowering byte extracts can leave widen(narrow(x)) behind again */
   NIR_PASS(_, s, nir_shader_alu_pass, fold_narrow_conversion, nir_metadata_control_flow, NULL);
   NIR_PASS(_, s, nir_opt_combine_stores, nir_var_shader_out);
   etna_optimize_loop(s);

   if (v->shader->specs->vs_need_z_div)
      NIR_PASS(_, s, nir_lower_clip_halfz);

   /* lower pre-halti2 to float (halti0 has integers, but only scalar..) */
   if (c->info->halti < 2) {
      /* use opt_algebraic between int_to_float and boot_to_float because
       * int_to_float emits ftrunc, and ftrunc lowering generates bool ops
       */
      NIR_PASS(_, s, nir_lower_int_to_float);
      NIR_PASS(_, s, nir_opt_algebraic);
      NIR_PASS(_, s, nir_lower_bool_to_float, true);
   } else {
      NIR_PASS(_, s, nir_lower_bool_to_int32);
   }

   while( OPT(s, nir_opt_vectorize, etna_vectorize_cb, NULL) );
   NIR_PASS(_, s, nir_lower_alu_to_scalar, etna_alu_to_scalar_filter_cb, c->info);

   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, s, nir_opt_algebraic_late);
   if (mesa_shader_stage_is_compute(s->info.stage))
      NIR_PASS(_, s, etna_narrow_products);
   /* the late rules fuse iadd(imul) into an imad as wide as the iadd, and
    * IMULLO0 and IMADLO0 are scalar: split the products now (lane products
    * compute every component) */
   NIR_PASS(_, s, nir_lower_alu_to_scalar, etna_alu_to_scalar_late_filter_cb, c->info);

   bool folded = false;
   NIR_PASS(folded, s, nir_shader_intrinsics_pass, fold_offset_shift, nir_metadata_control_flow, NULL);
   NIR_PASS(folded, s, nir_shader_intrinsics_pass, drop_narrow_store_mask, nir_metadata_control_flow, NULL);
   if (folded)
      NIR_PASS(_, s, nir_opt_dce);

   /* private arrays of 8- and 16-bit values (OpenCL) only reach SSA in the
    * optimization loop: lower the narrow values left, as in compile_shader() */
   NIR_PASS(_, s, nir_shader_instructions_pass, widen_16bit_image_access,
            nir_metadata_control_flow, NULL);
   for (unsigned i = 0; i < 8 && !v->dual16; i++) {
      bool narrow = false;
      NIR_PASS(narrow, s, nir_lower_bit_size, etna_lower_bit_size_cb, NULL);
      NIR_PASS(narrow, s, carry_half_phis_as_float);
      NIR_PASS(narrow, s, fold_half_round_trips);
      NIR_PASS(narrow, s, nir_opt_dce);
      NIR_PASS(narrow, s, nir_shader_alu_pass, lower_fp16_conversion, nir_metadata_none, NULL);
      NIR_PASS(narrow, s, nir_lower_fp16_casts, nir_lower_fp16_all);
      NIR_PASS(narrow, s, nir_shader_alu_pass, fold_narrow_conversion, nir_metadata_control_flow, NULL);
      if (!narrow)
         break;
      NIR_PASS(_, s, nir_opt_constant_folding);
      NIR_PASS(_, s, nir_opt_copy_prop);
      NIR_PASS(_, s, nir_opt_dce);
   }

   NIR_PASS(_, s, nir_move_vec_src_uses_to_dest, false);
   NIR_PASS(_, s, nir_opt_copy_prop);
   /* need copy prop after uses_to_dest, and before src mods: see
    * dEQP-GLES2.functional.shaders.random.all_features.fragment.95
    */

   NIR_PASS(_, s, nir_opt_dce);
   NIR_PASS(_, s, nir_opt_cse);

   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_opaque_copy, nir_metadata_control_flow, NULL);
   NIR_PASS(_, s, etna_nir_lower_bitfield_insert);
   NIR_PASS(_, s, nir_lower_bool_to_int32);
   NIR_PASS(_, s, etna_lower_alu, etna_core_has_feature(c->info, ETNA_FEATURE_HAS_FAST_TRANSCENDENTALS));
   if (c->info->halti >= 2) {
      bool sel = false;
      NIR_PASS(sel, s, nir_shader_alu_pass, fold_select_test, nir_metadata_control_flow, NULL);
      if (sel)
         NIR_PASS(_, s, nir_opt_dce);
   }

   if (spill_constants)
      NIR_PASS(_, s, etna_nir_spill_constants);

   /* needs to be the last pass that touches pass_flags! */
   NIR_PASS(_, s, etna_nir_lower_to_source_mods);

   if (v->dual16) {
      if (!dual16_check(c, s, true)) {
         ralloc_free(c->nir);
         free(c->code);
         FREE(c);
         return false;
      }
      c->d16 = rzalloc_array(c->nir, struct etna_d16_inst, ETNA_MAX_SHADER_INSTRUCTIONS);
   }

   if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS))
      nir_print_shader(s, stdout);

   unsigned block_ptr[nir_shader_get_entrypoint(s)->num_blocks];
   c->block_ptr = block_ptr;

   unsigned num_consts;
   bool ok = emit_shader(c, &v->num_temps, &num_consts);
   if (ok && v->dual16)
      ok = dual16_rewrite(c, v, ARRAY_SIZE(block_ptr));
   if (!ok || num_consts > max_uniforms(v)) {
      ralloc_free(c->nir);
      free(c->code);
      FREE(c);
      return false;
   }

   /* empty shader, emit NOP */
   if (!c->inst_ptr)
      emit_inst(c, &(struct etna_inst) { .opcode = ISA_OPC_NOP });

   /* assemble instructions, fixing up labels */
   const bool has_no_oneconst_limit = etna_core_has_feature(c->info, ETNA_FEATURE_SH_NO_ONECONST_LIMIT);
   uint32_t *code = MALLOC(c->inst_ptr * 16);
   v->cs_has_loops = false;
   for (unsigned i = 0; i < c->inst_ptr; i++) {
      struct etna_inst *inst = &c->code[i];
      if (inst->opcode == ISA_OPC_BRANCH || inst->opcode == ISA_OPC_BRANCH_UNARY || inst->opcode == ISA_OPC_BRANCH_BINARY) {
         inst->imm = block_ptr[inst->imm];
         v->cs_has_loops |= inst->imm <= i;
      }

      etna_assemble(&code[i * 4], inst, has_no_oneconst_limit);
   }

   v->code_size = c->inst_ptr * 4;
   v->code = code;
   v->needs_icache = c->inst_ptr > specs->max_instructions;

   copy_uniform_state_to_shader(v, c->consts, num_consts);

   if (s->info.stage == MESA_SHADER_FRAGMENT) {
      v->input_count_unk8 = 31; /* XXX what is this */
      assert(v->ps_depth_out_reg <= 0);
   } else if (s->info.stage == MESA_SHADER_VERTEX) {
      fill_vs_mystery(v);
   }

   if (s->constant_data_size) {
      v->constant_data = MALLOC(s->constant_data_size);
      memcpy(v->constant_data, s->constant_data, s->constant_data_size);
      v->constant_data_size = s->constant_data_size;
   }

   bool result = etna_compile_check_limits(v);
   ralloc_free(c->nir);
   free(c->code);
   FREE(c);
   return result;
}

static bool
compile_shader_spill(struct etna_shader_variant *v, unsigned fold, bool fold_loop)
{
   if (compile_shader(v, false, fold, fold_loop))
      return true;

   /* the spilled constants are read with LOAD, which needs halti2 */
   if (!v || v->shader->info->halti < 2)
      return false;

   return compile_shader(v, true, fold, fold_loop);
}

/* A compute workgroup with barrier() must be resident in full: invocations x
 * temporaries <= 512 per shader core (GC7000 r6204). The boundary itself
 * passes alone but hangs under load (four dEQP processes), counting the
 * temporaries in pairs: 64 x 16 and 16 x 62 hung. So did 168 x 6 = 1008 for
 * an OpenCL copy kernel with two barriers, even with 100 workgroups, while
 * 160 x 6 passed with 1749: stay at 480 per core or below
 * (etna_cs_barrier_capacity()).
 * If it does not fit, run the workgroup with fewer hardware threads, several
 * invocations each (see etna_nir_fold_workgroup()). ETNA_CS_FOLD sets a
 * smallest fold factor, for testing. */
static bool
fits(const struct etna_shader_variant *v, unsigned invocations)
{
   const unsigned temps = align(MAX2(v->num_temps, v->cs_input_count), 2);

   return DIV_ROUND_UP(invocations, v->cs_fold) * temps <= etna_cs_barrier_capacity(v->shader->info);
}

bool
etna_compile_shader(struct etna_shader_variant *v)
{
   if (!v)
      return false;

   const struct shader_info *info = &v->shader->nir->info;
   /* a kernel of variable workgroup size is folded in a variant for the
    * dispatch's size (key cs_block) */
   const uint16_t *wg = v->key.cs_block[0] ? v->key.cs_block : info->workgroup_size;
   const unsigned n = wg[0] * wg[1] * wg[2];
   const unsigned min_fold = info->stage == MESA_SHADER_COMPUTE ? debug_get_num_option("ETNA_CS_FOLD", 1) : 1;

   const bool foldable = info->stage == MESA_SHADER_COMPUTE && info->uses_control_barrier &&
                         (!info->workgroup_size_variable || v->key.cs_block[0]);

   unsigned unfolded_size = 0;

   if (!foldable || min_fold <= 1) {
      if (!compile_shader_spill(v, 1, false))
         return false;
      if (!foldable || fits(v, n))
         return true;
      unfolded_size = v->code_size / 4;
   }

   /* At least 32 threads: the folded code grows with the fold factor, in
    * size and in temporaries, and a 16-thread variant of 2600 instructions
    * still hung under load. A fixed workgroup size (OpenGL ES) takes the
    * powers of two that divide it.
    * A variant for the workgroup size of a dispatch (key cs_block) has to
    * run any size up to the one etna_get_compute_state_info() reports:
    * any fold factor, from the estimate of the temporaries of the last
    * compile, and down to 16 threads (a workgroup just above what fits
    * unfolded folds by 2). */
   const bool any_fold = v->key.cs_block[0] != 0;
   const unsigned capacity = etna_cs_barrier_capacity(v->shader->info);
   const unsigned min_threads = any_fold ? 16 : 32;
   unsigned fold = MAX2(min_fold, 2);

   if (any_fold && v->num_temps)
      fold = MAX2(fold, DIV_ROUND_UP(n * align(MAX2(v->num_temps, v->cs_input_count), 2), capacity));

   while (fold <= n / min_threads) {
      if (!any_fold && n % fold)
         break;

      /* Unrolled, the copies make the code F times as large, and
       * past the instruction cache (512 instructions on GC7000 r6204) every
       * thread streams it from memory: run them in a loop instead when the
       * unrolled code would be over twice that. Below, the unrolled copies
       * are faster, as they share the work that doesn't depend on the
       * invocation (OpenCV's GaussianBlur row filter, 184 instructions folded
       * by 3: 431 ms unrolled, 459 ms in a loop; Canny's stage1_with_sobel,
       * 438 instructions folded by 8: 810 ms unrolled, 618 ms in a loop,
       * folded by 4). */
      const bool compiled = compile_shader_spill(v, fold, fold * unfolded_size > 1024);
      if (compiled && fits(v, n))
         return true;

      if (!any_fold)
         fold *= 2;
      else if (!compiled)
         break; /* more copies only make it larger */
      else
         fold = MAX2(fold + 1, DIV_ROUND_UP(n * align(MAX2(v->num_temps, v->cs_input_count), 2), capacity));
   }

   /* nothing fits: the dispatch refuses it, see etna_launch_grid() */
   return compile_shader_spill(v, 1, false);
}

static const struct etna_shader_inout *
etna_shader_vs_lookup(const struct etna_shader_variant *sobj,
                      const struct etna_shader_inout *in)
{
   for (int i = 0; i < sobj->outfile.num_reg; i++)
      if (sobj->outfile.reg[i].slot == in->slot)
         return &sobj->outfile.reg[i];

   /*
    * There are valid NIR shaders pairs where the vertex shader has
    * a VARYING_SLOT_BFC0 shader_out and the corresponding framgent
    * shader has a VARYING_SLOT_COL0 shader_in.
    * So at link time if there is no matching VARYING_SLOT_BFC[n],
    * we must map VARYING_SLOT_BFC0[n] to VARYING_SLOT_COL[n].
    */
   gl_varying_slot slot;

   if (in->slot == VARYING_SLOT_COL0)
      slot = VARYING_SLOT_BFC0;
   else if (in->slot == VARYING_SLOT_COL1)
      slot = VARYING_SLOT_BFC1;
   else
      return NULL;

   for (int i = 0; i < sobj->outfile.num_reg; i++)
      if (sobj->outfile.reg[i].slot == slot)
         return &sobj->outfile.reg[i];

   return NULL;
}

void
etna_link_shader(struct etna_shader_link_info *info,
                 const struct etna_shader_variant *vs,
                 const struct etna_shader_variant *fs)
{
   int comp_ofs = 0;
   /* For each fragment input we need to find the associated vertex shader
    * output, which can be found by matching on semantic name and index. A
    * binary search could be used because the vs outputs are sorted by their
    * semantic index and grouped by semantic type by fill_in_vs_outputs.
    */
   assert(fs->infile.num_reg <= ETNA_NUM_INPUTS);
   info->pcoord_varying_comp_ofs = -1;

   for (int idx = 0; idx < fs->infile.num_reg; ++idx) {
      const struct etna_shader_inout *fsio = &fs->infile.reg[idx];
      const struct etna_shader_inout *vsio = etna_shader_vs_lookup(vs, fsio);
      struct etna_varying *varying;
      bool varying_is_color = fsio->slot == VARYING_SLOT_COL0 ||
                              fsio->slot == VARYING_SLOT_COL1;

      assert(fsio->reg > 0 && fsio->reg <= ARRAY_SIZE(info->varyings));

      if (fsio->reg > info->num_varyings)
         info->num_varyings = fsio->reg;

      varying = &info->varyings[fsio->reg - 1];
      varying->num_components = fsio->num_components;

      if (varying_is_color) /* colors affected by flat shading */
         varying->pa_attributes = 0x200;
      else /* texture coord or other bypasses flat shading */
         varying->pa_attributes = 0x2f1;

      for (int i = 0; i < 4; i++) {
         if (varying_is_color)
            varying->use[i] = VARYING_COMPONENT_USE_COLOR;
         else
            varying->use[i] = VARYING_COMPONENT_USE_GENERIC;
      }

      switch (fsio->interpolation) {
      case INTERP_MODE_NONE:
      case INTERP_MODE_SMOOTH:
         varying->semantic = VARYING_INTERPOLATION_MODE_SMOOTH;
         break;
      case INTERP_MODE_NOPERSPECTIVE:
         varying->semantic = VARYING_INTERPOLATION_MODE_NONPERSPECTIVE;
         break;
      case INTERP_MODE_FLAT:
         varying->semantic = VARYING_INTERPOLATION_MODE_FLAT;
         break;
      default:
         UNREACHABLE("unsupported varying interpolation mode");
      }

      /* point/tex coord is an input to the PS without matching VS output,
       * so it gets a varying slot without being assigned a VS register.
       */
      if (fsio->slot == VARYING_SLOT_PNTC) {
         varying->use[0] = VARYING_COMPONENT_USE_POINTCOORD_X;
         varying->use[1] = VARYING_COMPONENT_USE_POINTCOORD_Y;

         info->pcoord_varying_comp_ofs = comp_ofs;
      } else if (util_varying_is_point_coord(fsio->slot, fs->key.sprite_coord_enable)) {
         /*
	  * Do nothing, TexCoord is lowered to PointCoord above
	  * and the TexCoord here is just a remnant. This needs
	  * to be removed with some nir_remove_dead_variables(),
	  * but that one removes all FS inputs ... why?
	  */
      } else {
         /* pick a random register to use if there is no VS output */
         if (vsio == NULL)
            varying->reg = 0;
         else
            varying->reg = vsio->reg;
      }

      comp_ofs += varying->num_components;
   }

   assert(info->num_varyings == fs->infile.num_reg);
}
