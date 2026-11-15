/*
 * Copyright © 2023 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "etnaviv_nir.h"
#include "nir_builtin_builder.h"

static bool
lower_txs(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txs)
      return false;

   b->cursor = nir_instr_remove(&tex->instr);

   nir_def *idx = nir_imm_int(b, tex->texture_index);
   nir_def *sizes = nir_load_texture_size_etna(b, 32, idx);

   nir_def_rewrite_uses(&tex->def, sizes);

   return true;
}

static nir_def *
clamp_lod(nir_builder *b, nir_def *sampler, nir_def *lod)
{
   nir_def *params = nir_load_sampler_lod_parameters(b, 2, 32, sampler);
   nir_def *min_lod = nir_channel(b, params, 0);
   nir_def *max_lod = nir_channel(b, params, 1);

   return nir_fclamp(b, lod, min_lod, max_lod);
}

static nir_def *
level_size(nir_builder *b, nir_def *base_size, nir_def *lod)
{
   /* Mip levels are never smaller than 1x1 */
   return nir_imax(b, nir_ushr(b, base_size, lod), nir_imm_int(b, 1));
}

static nir_def *
calculate_coord(nir_builder *b, nir_tex_instr *tex, nir_def *coord, nir_def *base_size_int, nir_def *lod, nir_def *offset)
{
   nir_def *mip_size = level_size(b, base_size_int, nir_f2i32(b, lod));

   /* Convert mip size to float and calculate reciprocal to scale texel offsets into normalized coordinates */
   nir_def *mip_size_float = nir_i2f32(b, mip_size);
   nir_def *inv_mip_size = nir_frcp(b, mip_size_float);

   offset = nir_i2f32(b, offset);

   if (tex->is_array) {
      const unsigned array_index = tex->coord_components - 1;

      /* Split coordinate into spatial part and array layer */
      nir_def *spatial_coord = nir_trim_vector(b, coord, array_index);
      nir_def *array_layer = nir_channel(b, coord, array_index);

      /* Apply offset only to spatial coordinates */
      nir_def *spatial_inv_level_size = nir_trim_vector(b, inv_mip_size, array_index);
      spatial_coord = nir_fadd(b, spatial_coord,
                              nir_fmul(b, nir_trim_vector(b, offset, array_index),
                                       spatial_inv_level_size));

      /* Reconstruct full coordinate with original array layer */
      coord = nir_vec3(b, nir_channel(b, spatial_coord, 0),
                        nir_channel(b, spatial_coord, 1),
                        array_layer);
   } else {
      coord = nir_fadd(b, coord, nir_fmul(b, offset, inv_mip_size));
   }

   return coord;
}

static bool
lower_tex_offset(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_tex)
      return false;

   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);
   if (!offset)
      return false;

   assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *sampler = nir_imm_int(b, tex->texture_index);
   nir_def *lod = NULL;

   /* Load the base level texture size as int */
   nir_def *base_size_int = nir_load_texture_size_etna(b, 32, sampler);
   base_size_int = nir_trim_vector(b, base_size_int, tex->coord_components);

   nir_def *base_size = nir_i2f32(b, base_size_int);

   /* Compute texture coordinate derivatives */
   nir_def *ddx = nir_ddx(b, coord);
   nir_def *ddy = nir_ddy(b, coord);

   /* Scale derivatives by texture size */
   nir_def *scaled_ddx = nir_fmul(b, ddx, base_size);
   nir_def *scaled_ddy = nir_fmul(b, ddy, base_size);

   /* Calculate LOD using scaled derivatives with fdot calls */
   nir_def *ddx_squared = nir_fdot(b, scaled_ddx, scaled_ddx);
   nir_def *ddy_squared = nir_fdot(b, scaled_ddy, scaled_ddy);

   nir_def *max_derivative = nir_fmax(b, ddx_squared, ddy_squared);

   /* Hardware-specific LOD quantization using IEEE 754 float manipulation.
    * By multiplying by 0.5 and adding 393216.0f (2^18 + 2^17), we force a
    * specific exponent that traps the fractional LOD bits in the mantissa.
    * This creates a float where the mantissa behaves like a 4.4 fixed-point
    * value, matching the expected behaviour of Vivante GPU.
    */
   nir_def *lod_raw = nir_flog2(b, max_derivative);
   nir_def *lod_fixed_point = nir_fmad(b, lod_raw, nir_imm_float(b, 0.5f),
                                       nir_imm_float(b, 393216.0f));

   /* Extract 16-bit fractional part */
   nir_def *lod_masked = nir_iand_imm(b, lod_fixed_point, 0xFFFF);

   /* Handle sign extension for negative LODs */
   nir_def *sign_bit = nir_iand_imm(b, lod_masked, 0x8000);
   nir_def *lod_or_magic = nir_ior_imm(b, lod_masked, 0xFFFF0000);

   /* Select sign-extended version for negative, original for positive */
   nir_def *lod_quantized = nir_bcsel(b, nir_ine_imm(b, sign_bit, 0), lod_or_magic, lod_masked);

   /* Convert back from fixed-point: scale by 1/32 and add 0.5 offset
    * This reverses the fixed-point encoding to get final LOD value
    */
   nir_def *lod_float = nir_u2f32(b, lod_quantized);
   lod = nir_fmad(b, lod_float, nir_imm_float(b, 1.0f/32.0f), nir_imm_float(b, 0.5f));

   /* floor and convert to int */
   lod = nir_ffloor(b, lod);
   lod = clamp_lod(b, sampler, lod);

   coord = calculate_coord(b, tex, coord, base_size_int, lod, offset);

   nir_src_rewrite(&tex->src[coord_index].src, coord);

   return true;
}

static bool
lower_txl_offset(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txl)
      return false;

   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);
   if (!offset)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *lod = nir_get_tex_src(tex, nir_tex_src_lod);

   assert(coord_index >= 0);
   assert(lod);

   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *sampler = nir_imm_int(b, tex->texture_index);

   nir_def *base_size_int = nir_load_texture_size_etna(b, 32, sampler);
   base_size_int = nir_trim_vector(b, base_size_int, tex->coord_components);

   /* Round LOD to nearest integer using floor(lod + 0.5) */
   lod = nir_fadd_imm(b, lod, 0.5f);
   lod = nir_ffloor(b, lod);
   lod = clamp_lod(b, sampler, lod);

   coord = calculate_coord(b, tex, coord, base_size_int, lod, offset);

   nir_src_rewrite(&tex->src[coord_index].src, coord);

   return true;
}

static bool
lower_txd_offset(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txd)
      return false;

   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);
   if (!offset)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int ddx_index = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   int ddy_index = nir_tex_instr_src_index(tex, nir_tex_src_ddy);

   assert(coord_index >= 0);
   assert(ddx_index >= 0);
   assert(ddy_index >= 0);

   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *ddx = tex->src[ddx_index].src.ssa;
   nir_def *ddy = tex->src[ddy_index].src.ssa;
   nir_def *sampler = nir_imm_int(b, tex->texture_index);

   /* Load the base level texture size and convert to float for gradient scaling */
   nir_def *base_size_int = nir_load_texture_size_etna(b, 32, sampler);
   base_size_int = nir_trim_vector(b, base_size_int, tex->coord_components);
   nir_def *base_size_float = nir_i2f32(b, base_size_int);

   /* Scale gradients from normalized space to texel space */
   nir_def *scaled_ddx = nir_fmul(b, ddx, base_size_float);
   nir_def *scaled_ddy = nir_fmul(b, ddy, base_size_float);

   /* Compute the absolute values of scaled gradients */
   nir_def *abs_ddx = nir_fabs(b, scaled_ddx);
   nir_def *abs_ddy = nir_fabs(b, scaled_ddy);

   /* Take the component-wise maximum of |ddx| and |ddy| */
   nir_def *max_grad = nir_fmax(b, abs_ddx, abs_ddy);

   /* Reduce to scalar max (for 2D: max(x,y); for 3D: max(x,y,z)) */
   nir_def *max_grad_scalar;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_3D) {
      nir_def *max_xy = nir_fmax(b, nir_channel(b, max_grad, 0),
                                    nir_channel(b, max_grad, 1));
      max_grad_scalar = nir_fmax(b, max_xy, nir_channel(b, max_grad, 2));
   } else {
      max_grad_scalar = nir_fmax(b, nir_channel(b, max_grad, 0),
                                    nir_channel(b, max_grad, 1));
   }

   /* Compute log2(max_grad) for LOD */
   nir_def *lod = nir_flog2(b, max_grad_scalar);

   /* Round LOD to nearest integer using floor(lod + 0.5) */
   lod = nir_fadd_imm(b, lod, 0.5f);
   lod = nir_ffloor(b, lod);
   lod = clamp_lod(b, sampler, lod);

   coord = calculate_coord(b, tex, coord, base_size_int, lod, offset);

   nir_src_rewrite(&tex->src[coord_index].src, coord);

   return true;
}

static bool
lower_tg4_offset(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_tg4)
      return false;

   nir_def *offset = nir_steal_tex_src(tex, nir_tex_src_offset);
   if (!offset)
      return false;

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   assert(coord_index >= 0);
   nir_def *coord = tex->src[coord_index].src.ssa;

   b->cursor = nir_before_instr(&tex->instr);

   /* Clamp offset to min_texel_offset and max_texel_offset */
   offset = nir_iclamp(b, offset, nir_imm_int(b, -8), nir_imm_int(b, 7));

   nir_def *sampler = nir_imm_int(b, tex->texture_index);
   nir_def *base_size_int = nir_load_texture_size_etna(b, 32, sampler);
   base_size_int = nir_trim_vector(b, base_size_int, tex->coord_components);

   nir_def *base_size = nir_i2f32(b, base_size_int);
   offset = nir_i2f32(b, offset);

   if (tex->is_array) {
      const unsigned array_index = tex->coord_components - 1;

      /* Split coordinate into spatial part and array layer */
      nir_def *spatial_coord = nir_trim_vector(b, coord, array_index);
      nir_def *array_layer = nir_channel(b, coord, array_index);

      /* Apply offset only to spatial coordinates */
      spatial_coord = nir_fadd(b, spatial_coord,
                              nir_fdiv(b, nir_trim_vector(b, offset, array_index),
                                       base_size));

      /* Reconstruct full coordinate with original array layer */
      coord = nir_vec3(b, nir_channel(b, spatial_coord, 0),
                        nir_channel(b, spatial_coord, 1),
                        array_layer);
   } else {
      coord = nir_fadd(b, coord, nir_fdiv(b, offset, base_size));
   }

   nir_src_rewrite(&tex->src[coord_index].src, coord);

   return true;
}

static bool
lower_txf(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int lod_index = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   assert(coord_index >= 0);
   assert(lod_index >= 0);
   assert(nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0);

   nir_def *coord = nir_i2f32(b, tex->src[coord_index].src.ssa);
   nir_def *lod = tex->src[lod_index].src.ssa;
   nir_def *sampler = nir_imm_int(b, tex->texture_index);
   nir_def *size = nir_trim_vector(b, nir_load_texture_size_etna(b, 32, sampler),
                                   tex->coord_components);
   nir_def *mip_size = nir_i2f32(b, level_size(b, size, lod));
   nir_def *center = nir_fdiv(b, nir_fadd_imm(b, coord, 0.5f), mip_size);

   if (tex->is_array) {
      const unsigned array_index = tex->coord_components - 1;

      center = nir_vector_insert_imm(b, center, nir_channel(b, coord, array_index),
                                     array_index);
   }

   nir_src_rewrite(&tex->src[coord_index].src, center);
   nir_src_rewrite(&tex->src[lod_index].src, nir_i2f32(b, lod));
   tex->op = nir_texop_txl;

   return true;
}

static bool
clamp_txf_coords(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int lod_index = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   assert(coord_index >= 0);
   assert(lod_index >= 0);

   const unsigned spatial = tex->coord_components - tex->is_array;
   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_src *lod = &tex->src[lod_index].src;
   nir_def *sampler = nir_imm_int(b, tex->texture_index);
   nir_def *size = nir_load_texture_size_etna(b, 32, sampler);
   nir_def *limit = nir_trim_vector(b, size, spatial);

   if (!nir_src_is_const(*lod) || nir_src_as_uint(*lod) != 0)
      limit = level_size(b, limit, lod->ssa);

   nir_def *clamped = nir_imin(b, nir_trim_vector(b, coord, spatial),
                               nir_iadd_imm(b, limit, -1));

   if (tex->is_array)
      clamped = nir_vector_insert_imm(b, nir_pad_vector(b, clamped, spatial + 1),
                                      nir_channel(b, coord, spatial), spatial);

   nir_src_rewrite(&tex->src[coord_index].src, clamped);

   return true;
}

static bool
legalize_txf_lod(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int lod_index = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   assert(lod_index >= 0);
   nir_def *lod = tex->src[lod_index].src.ssa;

   nir_src_rewrite(&tex->src[lod_index].src, nir_i2f32(b, lod));

   return true;
}

static bool
lower_txf_ms_dynamic(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf_ms)
      return false;

   int ms_index = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   assert(ms_index >= 0);

   if (nir_src_is_const(tex->src[ms_index].src))
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *index = tex->src[ms_index].src.ssa;
   nir_def *fetches[ETNA_MAX_SAMPLES];

   for (unsigned s = 0; s < ARRAY_SIZE(fetches); s++) {
      nir_def *sample = nir_imm_int(b, s);
      nir_tex_instr *fetch =
         nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));

      nir_builder_instr_insert(b, &fetch->instr);
      nir_src_rewrite(&fetch->src[ms_index].src, sample);

      fetches[s] = &fetch->def;
   }

   nir_def_rewrite_uses(&tex->def,
                        nir_select_from_ssa_def_array(b, fetches,
                                                      ARRAY_SIZE(fetches),
                                                      index));
   nir_instr_remove(&tex->instr);

   return true;
}

static bool
legalize_txd_derivatives(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txd)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int ddx_index = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   int ddy_index = nir_tex_instr_src_index(tex, nir_tex_src_ddy);

   assert(coord_index >= 0);
   assert(ddx_index >= 0);
   assert(ddy_index >= 0);

   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *ddx = tex->src[ddx_index].src.ssa;
   nir_def *ddy = tex->src[ddy_index].src.ssa;

   coord = nir_trim_vector(b, coord, ddx->num_components);

   nir_src_rewrite(&tex->src[ddx_index].src, nir_fadd(b, coord, ddx));
   nir_src_rewrite(&tex->src[ddy_index].src, nir_fadd(b, coord, ddy));

   return true;
}

static bool
legalize_txd_comparator(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txd)
      return false;

   if (!tex->is_shadow)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *comp = nir_steal_tex_src(tex, nir_tex_src_comparator);
   assert(comp);

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   int ddx_index = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
   int ddy_index = nir_tex_instr_src_index(tex, nir_tex_src_ddy);

   assert(coord_index >= 0);
   assert(ddx_index >= 0);
   assert(ddy_index >= 0);

   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *ddx = tex->src[ddx_index].src.ssa;
   nir_def *ddy = tex->src[ddy_index].src.ssa;

   coord = nir_pad_vec4(b, coord);
   coord = nir_vector_insert_imm(b, coord, comp, 3);

   /* Make nir validation happy. */
   ddx = nir_pad_vector(b, ddx, tex->is_array ? 3 : 4);
   ddy = nir_pad_vector(b, ddy, tex->is_array ? 3 : 4);

   nir_src_rewrite(&tex->src[coord_index].src, coord);
   nir_src_rewrite(&tex->src[ddx_index].src, ddx);
   nir_src_rewrite(&tex->src[ddy_index].src, ddy);

   tex->coord_components = 4;

   return true;
}

static bool
legalize_src(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   int bias_index = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   int lod_index = nir_tex_instr_src_index(tex, nir_tex_src_lod);

   if (bias_index < 0 && lod_index < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *src1 = NULL;
   if (bias_index >= 0)
      src1 = nir_steal_tex_src(tex, nir_tex_src_bias);

   if (lod_index >= 0) {
      assert(!src1);
      src1 = nir_steal_tex_src(tex, nir_tex_src_lod);
   }

   int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   assert(coord_index >= 0);
   nir_def *coord = tex->src[coord_index].src.ssa;

   assert(src1->num_components == 1);
   assert(tex->coord_components < 4);
   coord = nir_pad_vec4(b, coord);
   coord = nir_vector_insert_imm(b, coord, src1, 3);

   tex->coord_components = 4;

   nir_src_rewrite(&tex->src[coord_index].src, coord);

   return true;
}

static bool
lower_offset_filter(const nir_instr *instr, const void *data)
{
   const struct shader_info *info = data;

   assert(instr->type == nir_instr_type_tex);
   nir_tex_instr *tex = nir_instr_as_tex(instr);

   if (tex->op == nir_texop_tex && info->stage == MESA_SHADER_VERTEX)
      return true;

   if (tex->op == nir_texop_txb)
      return true;

   if (tex->op == nir_texop_txf)
      return true;

   return false;
}

/* is component c of a coordinate an integer converted to float */
static bool
is_int_coord(nir_def *coord, unsigned c)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(coord, c));

   if (!nir_scalar_is_alu(s))
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   return op == nir_op_i2f32 || op == nir_op_u2f32;
}

/* Samplers with unnormalized coordinates (OpenCL) are in the key: the texture
 * unit takes normalized ones only, so divide by the base level size. An
 * integer coordinate (read_image*(image, sampler, int2)) names a texel: move
 * it to the texel centre first, so that nearest filtering cannot round it
 * down to the previous texel after the division. The array layer stays. */
static bool
lower_unnormalized_coords(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;
   const unsigned i = tex->texture_index;

   if (i >= 16 || !(key->tex_unnormalized & BITFIELD_BIT(i)))
      return false;

   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txd && tex->op != nir_texop_tg4)
      return false;

   const int idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (idx < 0)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *coord = tex->src[idx].src.ssa;
   const unsigned n = tex->coord_components - tex->is_array;
   nir_def *size = nir_i2f32(b, nir_trim_vector(b, nir_load_texture_size_etna(b, 32, nir_imm_int(b, i)), n));
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];

   for (unsigned c = 0; c < tex->coord_components; c++) {
      comps[c] = nir_channel(b, coord, c);
      if (c >= n)
         continue;
      if (is_int_coord(coord, c))
         comps[c] = nir_fadd_imm(b, comps[c], 0.5f);
      comps[c] = nir_fdiv(b, comps[c], nir_channel(b, size, c));
   }

   nir_src_rewrite(&tex->src[idx].src, nir_vec(b, comps, tex->coord_components));
   return true;
}

/* 8-bit normalized texels convert to float slightly off in the texture unit
 * (see etna_launch_grid()): a read that doesn't filter, a fetch or a nearest
 * sample, snaps to the nearest k / 255 (k / 127 for SNORM). The error is far
 * below half a step. */
static bool
snap_norm8(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;
   const unsigned i = tex->texture_index;

   if (i >= 16 || !((key->tex_unorm8 | key->tex_snorm8) & BITFIELD_BIT(i)) ||
       nir_alu_type_get_base_type(tex->dest_type) != nir_type_float)
      return false;

   const bool fetch = tex->op == nir_texop_txf || tex->op == nir_texop_tg4;
   const bool sample = tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
                       tex->op == nir_texop_txl || tex->op == nir_texop_txd;

   if (!fetch && !(sample && (key->tex_nearest & BITFIELD_BIT(i))))
      return false;

   const float steps = (key->tex_unorm8 & BITFIELD_BIT(i)) ? 255.0f : 127.0f;

   b->cursor = nir_after_instr(&tex->instr);

   nir_def *v = nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, &tex->def, steps)), 1.0f / steps);
   if (key->tex_snorm8 & BITFIELD_BIT(i))
      v = nir_fmax_imm(b, v, -1.0f);

   nir_def_rewrite_uses_after(&tex->def, v);
   return true;
}

/* one nearest tap of an RGBA16 UNORM texture: the texel as 16-bit unsigned
 * integers, converted to float. coord NULL keeps the instruction's. */
static nir_def *
unorm16_tap(nir_builder *b, nir_tex_instr *tex, nir_def *coord)
{
   nir_tex_instr *t = nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));

   t->dest_type = nir_type_uint32;
   /* the clone joins the use lists of its sources on insertion */
   nir_builder_instr_insert(b, &t->instr);
   if (coord)
      nir_src_rewrite(&t->src[nir_tex_instr_src_index(t, nir_tex_src_coord)].src, coord);

   return nir_fmul_imm(b, nir_u2f32(b, &t->def), 1.0f / 65535.0f);
}

/* RGBA16 UNORM (OpenCL) has no texture format: the texture unit samples it as
 * 16-bit unsigned integers with a nearest sampler (etna_compute_swap_textures)
 * and the shader converts. A linear sampler filters here, from the nearest
 * taps around the coordinate, which the texture unit wraps or clamps as the
 * sampler says. Runs after lower_unnormalized_coords(). */
static bool
lower_unorm16(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;
   const unsigned i = tex->texture_index;

   if (i >= 16 || !(key->tex_unorm16 & BITFIELD_BIT(i)) ||
       nir_alu_type_get_base_type(tex->dest_type) != nir_type_float)
      return false;

   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txd && tex->op != nir_texop_txf)
      return false;

   b->cursor = nir_after_instr(&tex->instr);

   const int ci = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *v = NULL;

   if (tex->op == nir_texop_txf || ci < 0 || !(key->tex_unorm16_linear & BITFIELD_BIT(i))) {
      v = unorm16_tap(b, tex, NULL);
   } else {
      nir_def *coord = tex->src[ci].src.ssa;
      const unsigned n = tex->coord_components - tex->is_array;
      nir_def *size = nir_i2f32(b, nir_trim_vector(b, nir_load_texture_size_etna(b, 32, nir_imm_int(b, i)), n));
      nir_def *lo[3], *hi[3], *frac[3];

      for (unsigned c = 0; c < n; c++) {
         nir_def *s = nir_channel(b, size, c);
         nir_def *x = nir_fadd_imm(b, nir_fmul(b, nir_channel(b, coord, c), s), -0.5f);
         nir_def *x0 = nir_ffloor(b, x);

         frac[c] = nir_fsub(b, x, x0);
         lo[c] = nir_fdiv(b, nir_fadd_imm(b, x0, 0.5f), s);
         hi[c] = nir_fdiv(b, nir_fadd_imm(b, x0, 1.5f), s);
      }

      for (unsigned m = 0; m < (1u << n); m++) {
         nir_def *comps[4];
         nir_def *w = nir_imm_float(b, 1.0f);

         for (unsigned c = 0; c < tex->coord_components; c++) {
            if (c >= n) {
               comps[c] = nir_channel(b, coord, c);
               continue;
            }
            const bool up = m & BITFIELD_BIT(c);
            comps[c] = up ? hi[c] : lo[c];
            w = nir_fmul(b, w, up ? frac[c] : nir_fsub_imm(b, 1.0f, frac[c]));
         }

         nir_def *tap = nir_fmul(b, unorm16_tap(b, tex, nir_vec(b, comps, tex->coord_components)), w);
         v = v ? nir_fadd(b, v, tap) : tap;
      }
   }

   nir_def_replace(&tex->def, v);
   return true;
}

/* Emulated depth32f is D24 storage, so the texture unit clamps the border
 * depth to [0, 1]. A float depth border outside that range is in the shader
 * key (tex_border_depth), and the shader compares border texels with it.
 */
static bool
has_float_border(const struct etna_shader_key *key, const nir_tex_instr *tex)
{
   const unsigned i = tex->texture_index;

   return i < 16 && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
          ((key->tex_float_border_s | key->tex_float_border_t) & BITFIELD_BIT(i));
}

/* texel (x, y) of the base level, xy as ivec2: is it a border texel */
static nir_def *
is_border_texel(nir_builder *b, const struct etna_shader_key *key, unsigned i,
                nir_def *size, nir_def *xy)
{
   nir_def *border = nir_imm_false(b);

   for (unsigned c = 0; c < 2; c++) {
      if (!((c ? key->tex_float_border_t : key->tex_float_border_s) & BITFIELD_BIT(i)))
         continue;

      nir_def *v = nir_channel(b, xy, c);
      border = nir_ior(b, border, nir_ior(b, nir_ilt_imm(b, v, 0),
                                          nir_ige(b, v, nir_channel(b, size, c))));
   }

   return border;
}

/* base level texel of a 2D coordinate, plus the texture offset */
static nir_def *
base_texel(nir_builder *b, nir_tex_instr *tex, nir_def *size, float bias)
{
   nir_def *uv = nir_channels(b, nir_get_tex_src(tex, nir_tex_src_coord), 0x3);
   nir_def *p = nir_fadd_imm(b, nir_fmul(b, uv, nir_i2f32(b, nir_trim_vector(b, size, 2))), bias);
   nir_def *xy = nir_f2i32(b, nir_ffloor(b, p));
   nir_def *offset = nir_get_tex_src(tex, nir_tex_src_offset);

   return offset ? nir_iadd(b, xy, nir_trim_vector(b, offset, 2)) : xy;
}

static nir_def *
border_compare(nir_builder *b, const struct etna_shader_key *key, unsigned i, nir_def *ref)
{
   enum compare_func func = i < key->num_texture_states ?
      key->tex_compare_func[i] : COMPARE_FUNC_ALWAYS;

   return nir_b2f32(b, nir_compare_func(b, func, ref, nir_imm_float(b, key->tex_border_depth[i])));
}

/* shadow texture() on a float border sampler: a border texel compares with
 * the key's border depth. Runs before nir_lower_tex_shadow(), which turns the
 * result used here into the compare of the sampled depth. */
static bool
lower_float_border_tex(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;

   if (!tex->is_shadow || tex->def.num_components != 1 || !has_float_border(key, tex) ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl))
      return false;

   nir_def *ref = nir_get_tex_src(tex, nir_tex_src_comparator);
   if (!ref || !nir_get_tex_src(tex, nir_tex_src_coord))
      return false;

   const unsigned i = tex->texture_index;
   b->cursor = nir_after_instr(&tex->instr);

   nir_def *size = nir_load_texture_size_etna(b, 32, nir_imm_int(b, i));
   nir_def *border = is_border_texel(b, key, i, size, base_texel(b, tex, size, 0.0f));
   nir_def *res = nir_bcsel(b, border, border_compare(b, key, i, ref), &tex->def);

   nir_def_rewrite_uses_after(&tex->def, res);
   return true;
}

/* nir_lower_tex_shadow() leaves gathers alone, and the TG4 compare doesn't
 * work: compare each of the four gathered texels in the shader. */
static bool
lower_tg4_shadow(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct etna_shader_key *key = data;

   if (tex->op != nir_texop_tg4 || !tex->is_shadow)
      return false;

   int comp_index = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   assert(comp_index >= 0);
   nir_def *ref = tex->src[comp_index].src.ssa;

   tex->is_shadow = false;
   tex->component = 0;
   nir_tex_instr_remove_src(tex, comp_index);

   b->cursor = nir_after_instr(&tex->instr);

   if (!key->shadow_compare_no_clamp)
      ref = nir_fsat(b, ref);

   enum compare_func func = tex->texture_index < key->num_texture_states ?
      key->tex_compare_func[tex->texture_index] : COMPARE_FUNC_ALWAYS;

   /* the footprint is texels i0..i0+1, j0..j0+1, gathered in the order
    * (i0, j1), (i1, j1), (i1, j0), (i0, j0) */
   nir_def *border[4] = { NULL };
   if (has_float_border(key, tex) && !nir_tex_instr_has_explicit_tg4_offsets(tex) &&
       nir_get_tex_src(tex, nir_tex_src_coord)) {
      const unsigned s = tex->texture_index;
      nir_def *size = nir_load_texture_size_etna(b, 32, nir_imm_int(b, s));
      nir_def *ij = base_texel(b, tex, size, -0.5f);
      static const int dx[4] = { 0, 1, 1, 0 }, dy[4] = { 1, 1, 0, 0 };

      for (unsigned i = 0; i < 4; i++)
         border[i] = is_border_texel(b, key, s, size,
                                     nir_iadd(b, ij, nir_imm_ivec2(b, dx[i], dy[i])));
   }

   nir_def *res[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *depth = nir_channel(b, &tex->def, i);
      if (border[i])
         depth = nir_bcsel(b, border[i], nir_imm_float(b, key->tex_border_depth[tex->texture_index]), depth);
      res[i] = nir_b2f32(b, nir_compare_func(b, func, ref, depth));
   }

   nir_def_rewrite_uses_after(&tex->def, nir_vec(b, res, 4));
   return true;
}

bool
etna_nir_lower_texture(nir_shader *s, struct etna_shader_key *key, const struct etna_core_info *info)
{
   bool progress = false;

   nir_lower_tex_options lower_tex_options = {
      .callback_data = &s->info,
      .lower_txp = ~0u,
      .lower_txs_lod = true,
      .lower_invalid_implicit_lod = true,
      .lower_offset_filter = lower_offset_filter,
      .swizzle_result = key->tex_is_128bit,
      .lower_txl_mag_switchover = key->tex_mag_switchover,
   };

   u_foreach_bit(i, key->tex_is_128bit) {
      lower_tex_options.swizzles[i][0] = key->tex_swizzle[i].swizzle_r;
      lower_tex_options.swizzles[i][1] = key->tex_swizzle[i].swizzle_g;
      lower_tex_options.swizzles[i][2] = key->tex_swizzle[i].swizzle_b;
      lower_tex_options.swizzles[i][3] = key->tex_swizzle[i].swizzle_a;
   }

   NIR_PASS(progress, s, nir_lower_tex, &lower_tex_options);

   if (key->tex_unnormalized)
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_unnormalized_coords,
               nir_metadata_control_flow, key);

   if (key->tex_unorm8 | key->tex_snorm8)
      NIR_PASS(progress, s, nir_shader_tex_pass, snap_norm8,
               nir_metadata_control_flow, key);

   if (key->tex_unorm16)
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_unorm16,
               nir_metadata_control_flow, key);

   if (key->tex_float_border_s | key->tex_float_border_t)
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_float_border_tex,
               nir_metadata_control_flow, key);

   if (key->has_sample_tex_compare)
      NIR_PASS(progress, s, nir_lower_tex_shadow, key->num_texture_states,
                                                  key->tex_compare_func,
                                                  key->tex_swizzle,
                                                  !key->shadow_compare_no_clamp);

   if (key->has_sample_tex_compare || key->has_tg4_compare)
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_tg4_shadow,
               nir_metadata_control_flow, key);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_txs,
         nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_tex_offset,
      nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_txl_offset,
      nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_txd_offset,
      nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_tg4_offset,
      nir_metadata_control_flow, NULL);

   if (etna_core_has_feature(info, ETNA_FEATURE_TX_INTEGER_COORDINATE_V2)) {
      NIR_PASS(progress, s, nir_shader_tex_pass, clamp_txf_coords,
         nir_metadata_control_flow, NULL);

      NIR_PASS(progress, s, nir_shader_tex_pass, legalize_txf_lod,
         nir_metadata_control_flow, NULL);
   } else {
      NIR_PASS(progress, s, nir_shader_tex_pass, lower_txf,
         nir_metadata_control_flow, NULL);
   }

   NIR_PASS(progress, s, nir_shader_tex_pass, lower_txf_ms_dynamic,
      nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, legalize_txd_derivatives,
      nir_metadata_control_flow, NULL);

   NIR_PASS(progress, s, nir_shader_tex_pass, legalize_txd_comparator,
      nir_metadata_control_flow, NULL);

   if (info->halti < 5) {
      NIR_PASS(progress, s, nir_shader_tex_pass, legalize_src,
         nir_metadata_control_flow, NULL);
   }

   return progress;
}
