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
 *    Christian Gmeiner <christian.gmeiner@gmail.com>
 */

#include "etnaviv_compiler.h"
#include "etnaviv_compiler_nir.h"
#include "etnaviv_debug.h"
#include "util/ralloc.h"

struct etna_compiler *
etna_compiler_create(const struct etna_core_info *info)
{
   struct etna_compiler *compiler = rzalloc(NULL, struct etna_compiler);
   bool has_sign_floor_ceil = etna_core_has_feature(info, ETNA_FEATURE_HAS_SIGN_FLOOR_CEIL);
   bool has_sin_cos_sqrt = etna_core_has_feature(info, ETNA_FEATURE_HAS_SQRT_TRIG);

   compiler->options = (nir_shader_compiler_options) {
      .has_texture_scaling = true,
      .lower_fpow = true,
      .lower_fround_even = true,
      .lower_ftrunc = true,
      .float_mul_add16 = nir_float_muladd_support_has_fmad | nir_float_muladd_support_fuse,
      .float_mul_add32 = nir_float_muladd_support_has_fmad | nir_float_muladd_support_fuse,
      .float_mul_add64 = nir_float_muladd_support_has_fmad | nir_float_muladd_support_fuse,
      .lower_mul_high = true,
      .lower_hadd = true,
      .lower_uadd_sat = true,
      .lower_usub_sat = true,
      .lower_iadd_sat = true,
      /* no 64-bit integers: OpenCL builtins and conversions can still bring
       * some (u2u64), split them into 32-bit halves */
      .lower_int64_options = (nir_lower_int64_options)~0,
      .lower_mul_2x32_64 = true,
      .lower_bitops = true,
      .lower_bitfield_insert = info->halti < 5,
      .lower_flrp32 = true,
      .lower_fmod = true,
      .lower_fdph = true,
      .lower_insert_byte = true,
      .lower_insert_word = true,
      .lower_fdiv = true, /* !specs->has_new_transcendentals */
      .lower_extract_byte = true,
      .lower_extract_word = true,
      .lower_fsign = !has_sign_floor_ceil,
      .lower_ffloor = !has_sign_floor_ceil,
      .lower_fceil = !has_sign_floor_ceil,
      .lower_fsqrt = !has_sin_cos_sqrt,
      .lower_sincos = !has_sin_cos_sqrt,
      .lower_uniforms_to_ubo = info->halti >= 2,
      .force_indirect_unrolling = nir_var_all,
      .max_unroll_iterations = 32,
      .max_samples = ETNA_MAX_SAMPLES,
      .lower_pack_32_2x16_split = true,
      /* the 64-bit packs stay: nir_lower_int64() splits 64-bit values into
       * them, and the optimizations cancel them out. Lowered, they turned
       * back into 64-bit arithmetic (u2u64) after the frontend's int64
       * lowering had run (OpenCL: libclc's software fma()) */
      .lower_pack_half_2x16 = true,
      .lower_pack_snorm_2x16 = true,
      .lower_pack_split = true,
      .lower_pack_unorm_2x16 = true,
      .lower_unpack_32_2x16_split = true,
      .lower_unpack_half_2x16 = true,
      .lower_unpack_snorm_2x16 = true,
      .lower_unpack_unorm_2x16 = true,
      .lower_find_lsb = true,
      .lower_ifind_msb = true,
      .lower_ufind_msb = true,
      .has_uclz = true,
      .has_cs_global_id = true,
      .no_integers = info->halti < 2,
   };

   /* ETNA_MESA_DEBUG=fast_fma, experimental and not conformant: claim a fused
    * multiply-add, so that OpenCL's fma() becomes the hardware's unfused
    * multiply-add instead of the exact software emulation (dozens of
    * instructions). libclc's own math then takes its hardware fma path too,
    * and rusticl reports CL_FP_FMA. fma() is no longer correctly rounded and
    * its product can overflow on its own: CTS bruteforce fma fails, for
    * example fma(-FLT_MAX, -FLT_MAX, -inf) is NaN, not -inf. For image work
    * where a difference in the last bit doesn't matter (OpenCV's warpAffine:
    * 13 fma() per pixel). */
   if (DBG_ENABLED(ETNA_DBG_FAST_FMA))
      compiler->options.float_mul_add32 |= nir_float_muladd_support_has_ffma;

   compiler->regs = etna_ra_setup(compiler);
   if (!compiler->regs) {
      ralloc_free((void *)compiler);
      compiler = NULL;
   }

   return compiler;
}

void
etna_compiler_destroy(const struct etna_compiler *compiler)
{
   ralloc_free((void *)compiler);
}

const nir_shader_compiler_options *
etna_compiler_get_options(struct etna_compiler *compiler)
{
   return &compiler->options;
}
