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
   case nir_op_imul:
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
   /* use inline immediates if possible */
   if (c->info->halti >= 2 && num_components == 1 &&
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
    * it instead of composing the destination swizzle onto it.
    */
   if (alu->op == nir_op_bitfield_insert_etna)
      srcs[2] = src_swizzle(get_src(c, &alu->src[2].src), SWIZZLE(X, Y, Y, Y));

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
      want[i] = CONST_VAL(ETNA_UNIFORM_IMAGE_DESC, image | plane << 4 | i << 8).u64;

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
      /* components of the format; 128-bit formats load two per plane */
      const unsigned comps = util_format_get_blocksize(format) == 16
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
      emit_inst(c, &(struct etna_inst) {
         .opcode = ISA_OPC_LOAD,
         .type = ISA_TYPE_U32,
         .dst = ra_def(c, &intr->def, &dst_swiz),
         .src[0] = get_src(c, &intr->src[1]),
         .src[1] = get_src(c, &intr->src[0]),
      });
   } break;
   case nir_intrinsic_store_global_etna:
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

static void
emit_instr(struct etna_compile *c, nir_instr * instr)
{
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
            case nir_intrinsic_load_num_workgroups:
               for (unsigned i = 0; i < intr->def.num_components; i++)
                  value[i] = CONST_VAL(ETNA_UNIFORM_NUM_WORKGROUPS, i);
               break;
            case nir_intrinsic_get_ssbo_size:
               value[0] = CONST_VAL(ETNA_UNIFORM_SSBO_SIZE, nir_src_as_uint(intr->src[0]));
               break;
            case nir_intrinsic_load_scratch_base_ptr:
               value[0] = CONST_VAL(ETNA_UNIFORM_SCRATCH_ADDR, 0);
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
            /* the address is a uniform: offset and data go through registers */
            const bool store = intr->intrinsic == nir_intrinsic_store_global_etna;
            const bool load = intr->intrinsic == nir_intrinsic_load_global_etna;
            nir_src *srcs[] = { store ? &intr->src[2] : &intr->src[1],
                                store ? &intr->src[0] : load ? NULL : &intr->src[2] };

            for (unsigned i = 0; i < ARRAY_SIZE(srcs); i++) {
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
         if (def && (def->bit_size == 8 || def->bit_size == 16) &&
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

   if (v->num_temps > info->gpu.max_registers) {
      DBG("Number of registers (%d) exceeds maximum %d", v->num_temps,
          info->gpu.max_registers);
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
      val = load_cs_input(b, 2, input_count);
      break;
   case nir_intrinsic_load_local_invocation_index: {
      const uint16_t *size = b->shader->info.workgroup_size;
      nir_def *id = load_cs_input(b, 1, input_count);

      val = nir_iadd(b, nir_channel(b, id, 0),
                     nir_imul_imm(b, nir_iadd(b, nir_channel(b, id, 1),
                                              nir_imul_imm(b, nir_channel(b, id, 2), size[1])),
                                  size[0]));
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&intr->def, nir_u2uN(b, val, intr->def.bit_size));
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
   nir_def *offset = nir_iadd(b, nir_imul_imm(b, group, slice),
                              nir_iadd_imm(b, intr->src[off_src].ssa, nir_intrinsic_base(intr)));
   nir_def *addr = nir_load_shared_base_ptr(b, 1, 32);
   nir_def *repl = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
      repl = nir_load_global_etna(b, intr->def.num_components, intr->def.bit_size, addr, offset);
      break;
   case nir_intrinsic_store_shared:
      nir_store_global_etna(b, intr->src[0].ssa, addr, offset,
                            .write_mask = nir_intrinsic_write_mask(intr));
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

/* Images are linear or 4x4 tiled (see etna_resource_create()). Loads and
 * stores use img_load/img_store, which address and convert texels through
 * the image descriptor. 128-bit formats are two stacked planes of 64-bit
 * texels (xy, then zw), see etna_resource_level_second_plane_offset().
 * Atomics (r32ui, r32i) address the texel in the shader and use the atomic
 * instructions: linear base + layer * layer stride + y * stride + x * 4,
 * tiled in tiles of 4x4 texels in rows of stride * 4 bytes. */
static bool
lower_image_access(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
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

   if (intr->intrinsic == nir_intrinsic_image_size) {
      nir_def *size[3] = {
         image_param(b, index, ETNA_IMAGE_WIDTH),
         image_param(b, index, ETNA_IMAGE_HEIGHT),
         image_param(b, index, ETNA_IMAGE_DEPTH),
      };
      if (dim == GLSL_SAMPLER_DIM_CUBE && array)
         size[2] = nir_udiv_imm(b, size[2], 6);
      nir_def_replace(&intr->def, nir_vec(b, size, intr->def.num_components));
      return true;
   }

   const enum pipe_format format = nir_intrinsic_format(intr);
   const unsigned bpp = util_format_get_blocksize(format);
   const bool planes = bpp == 16;
   const unsigned texel_bytes = planes ? 8 : bpp;
   nir_def *coord = intr->src[1].ssa;

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
                                          .image_dim = dim, .image_array = array, .format = format);
         if (planes) {
            nir_def *hi = nir_load_image_etna(b, index, hcoord[1], .component = 1,
                                              .image_dim = dim, .image_array = array, .format = format);
            t = nir_vec4(b, nir_channel(b, t, 0), nir_channel(b, t, 1),
                         nir_channel(b, hi, 0), nir_channel(b, hi, 1));
         } else if (util_format_get_nr_components(format) == 1) {
            /* the load writes .x only: (r, 0, 0, 1) */
            nir_def *one = util_format_is_pure_integer(format) ? nir_imm_int(b, 1)
                                                               : nir_imm_float(b, 1.0);
            t = nir_vec4(b, nir_channel(b, t, 0), nir_imm_int(b, 0), nir_imm_int(b, 0), one);
         }
         nir_def_replace(&intr->def, nir_trim_vector(b, t, intr->def.num_components));
      } else {
         nir_def *v = intr->src[3].ssa;
         nir_store_image_etna(b, index, hcoord[0], v, .component = 0,
                              .image_dim = dim, .image_array = array, .format = format);
         if (planes)
            nir_store_image_etna(b, index, hcoord[1], nir_swizzle(b, v, (unsigned[]){ 2, 3, 2, 3 }, 4),
                                 .component = 1, .image_dim = dim,
                                 .image_array = array, .format = format);
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

/* Memory is accessed in aligned 32-bit words: narrower or unaligned loads
 * become word loads plus extraction, narrower stores atomic and/or on the
 * word (OpenCL char and short data, 8-bit kernel arguments). */
static nir_mem_access_size_align
etna_mem_access_size_align(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size,
                           uint32_t align_mul, uint32_t align_offset, bool offset_is_const,
                           enum gl_access_qualifier access, const void *cb_data)
{
   return (nir_mem_access_size_align){
      .num_components = MIN2(DIV_ROUND_UP(bytes, 4), 4),
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
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

/* SSBO access becomes address + offset: the address is a uniform the driver
 * fills with the GPU address of the bound buffer */
static bool
lower_ssbo_access(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   nir_def *repl = NULL;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
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

static bool
compile_shader(struct etna_shader_variant *v, bool spill_constants, unsigned fold)
{
   if (unlikely(!v))
      return false;

   struct etna_compile *c = CALLOC_STRUCT(etna_compile);
   if (!c)
      return false;

   /* a retry starts from a clean variant */
   memset(VARIANT_CACHE_PTR(v), 0, VARIANT_CACHE_SIZE);

   c->variant = v;
   c->info = v->shader->info;
   c->specs = v->shader->specs;
   c->nir = nir_shader_clone(NULL, v->shader->nir);

   nir_shader *s = c->nir;
   const struct etna_specs *specs = c->specs;

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

   if (s->info.stage == MESA_SHADER_COMPUTE) {
      v->uses_discard = false;
      v->cs_fold = 1;

      if (fold > 1) {
         if (!etna_nir_fold_workgroup(s, fold, &v->cs_scratch_slots)) {
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
      if (OPT(s, nir_lower_mem_access_bit_sizes, &mem_opts)) {
         NIR_PASS(_, s, nir_lower_bit_size, etna_lower_bit_size_cb, NULL);
         NIR_PASS(_, s, nir_shader_alu_pass, fold_narrow_conversion,
                  nir_metadata_control_flow, NULL);
         NIR_PASS(_, s, nir_opt_constant_folding);
         NIR_PASS(_, s, nir_opt_algebraic);
         NIR_PASS(_, s, nir_opt_dce);
      }
      v->cs_shared_size = align(s->info.shared_size, 16);
      v->cs_uses_barrier = s->info.uses_control_barrier;
      if (v->cs_shared_size)
         NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_shared_access,
                  nir_metadata_none, &v->cs_shared_size);

      NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_cs_intrinsic,
               nir_metadata_control_flow, &v->cs_input_count);
   }

   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_image_access,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_ssbo_access,
            nir_metadata_control_flow, NULL);

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
   NIR_PASS(_, s, nir_lower_alu_width, alu_width_cb, NULL);

   NIR_PASS(_, s, nir_lower_alu_to_scalar, etna_alu_to_scalar_filter_cb, c->info);
   if (c->info->halti >= 2) {
      nir_lower_idiv_options idiv_options = {
         .allow_fp16 = true,
      };
      NIR_PASS(_, s, nir_lower_idiv, &idiv_options);
   }
   NIR_PASS(_, s, nir_lower_alu);

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

   while( OPT(s, nir_opt_vectorize, NULL, NULL) );
   NIR_PASS(_, s, nir_lower_alu_to_scalar, etna_alu_to_scalar_filter_cb, c->info);

   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, s, nir_opt_algebraic_late);
   NIR_PASS(_, s, nir_shader_alu_pass, fold_narrow_conversion, nir_metadata_control_flow, NULL);

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

   if (spill_constants)
      NIR_PASS(_, s, etna_nir_spill_constants);

   /* needs to be the last pass that touches pass_flags! */
   NIR_PASS(_, s, etna_nir_lower_to_source_mods);

   if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS))
      nir_print_shader(s, stdout);

   unsigned block_ptr[nir_shader_get_entrypoint(s)->num_blocks];
   c->block_ptr = block_ptr;

   unsigned num_consts;
   bool ok = emit_shader(c, &v->num_temps, &num_consts);
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
   for (unsigned i = 0; i < c->inst_ptr; i++) {
      struct etna_inst *inst = &c->code[i];
      if (inst->opcode == ISA_OPC_BRANCH || inst->opcode == ISA_OPC_BRANCH_UNARY || inst->opcode == ISA_OPC_BRANCH_BINARY)
         inst->imm = block_ptr[inst->imm];

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
compile_shader_spill(struct etna_shader_variant *v, unsigned fold)
{
   if (compile_shader(v, false, fold))
      return true;

   /* the spilled constants are read with LOAD, which needs halti2 */
   if (!v || v->shader->info->halti < 2)
      return false;

   return compile_shader(v, true, fold);
}

/* A compute workgroup with barrier() must be resident in full: invocations x
 * temporaries <= 512 per shader core (GC7000 r6204). The boundary itself
 * passes alone but hangs under load (four dEQP processes), counting the
 * temporaries in pairs: 64 x 16 and 16 x 62 hung. Stay below it.
 * If it does not fit, run the workgroup with fewer hardware threads, several
 * invocations each (see etna_nir_fold_workgroup()). ETNA_CS_FOLD sets a
 * smallest fold factor, for testing. */
static bool
fits(const struct etna_shader_variant *v, unsigned invocations)
{
   const unsigned temps = align(MAX2(v->num_temps, v->cs_input_count), 2);

   return invocations / v->cs_fold * temps < 512 * v->shader->info->gpu.shader_core_count;
}

bool
etna_compile_shader(struct etna_shader_variant *v)
{
   if (!v)
      return false;

   const struct shader_info *info = &v->shader->nir->info;
   const unsigned n = info->workgroup_size[0] * info->workgroup_size[1] * info->workgroup_size[2];
   const unsigned min_fold = info->stage == MESA_SHADER_COMPUTE ? debug_get_num_option("ETNA_CS_FOLD", 1) : 1;

   const bool foldable = info->stage == MESA_SHADER_COMPUTE && info->uses_control_barrier &&
                         !info->workgroup_size_variable;

   if (!foldable || min_fold <= 1) {
      if (!compile_shader_spill(v, 1))
         return false;
      if (!foldable || fits(v, n))
         return true;
   }

   /* at least 32 threads: the folded code grows with the fold factor, in
    * size and in temporaries, and a 16-thread variant of 2600 instructions
    * still hung under load */
   for (unsigned fold = MAX2(min_fold, 2); fold <= n / 32; fold *= 2) {
      if (n % fold)
         break;
      if (compile_shader_spill(v, fold) && fits(v, n))
         return true;
   }

   /* nothing fits: the dispatch refuses it, see etna_launch_grid() */
   return compile_shader_spill(v, 1);
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
