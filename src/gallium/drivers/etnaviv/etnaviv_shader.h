/*
 * Copyright (c) 2012-2015 Etnaviv Project
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
 *    Wladimir J. van der Laan <laanwj@gmail.com>
 */

#ifndef H_ETNAVIV_SHADER
#define H_ETNAVIV_SHADER

#include "mesa/main/config.h"
#include "etna_core_info.h"
#include "etnaviv_internal.h"
#include "nir.h"
#include "pipe/p_state.h"
#include "util/disk_cache.h"
#include "util/u_queue.h"
#include "util/u_shader_variant_cache.h"

struct etna_context;
struct etna_screen;
struct etna_shader_variant;
struct nir_shader;

struct etna_shader_key
{
   union {
      struct {
         /*
          * Combined Vertex/Fragment shader parameters:
          */

         /* do we need to invert front facing value? */
         unsigned front_ccw : 1;
         /* do we need to replace glTexCoord.xy ? */
         unsigned sprite_coord_enable : MAX_TEXTURE_COORD_UNITS;
         unsigned sprite_coord_yinvert : 1;
         /* do we need to lower sample_tex_compare */
         unsigned has_sample_tex_compare : 1;
         /* lowered shadow compare must not clamp the ref (float depth32f) */
         unsigned shadow_compare_no_clamp : 1;
         /* compare shadow gathers in the shader (the TG4 compare doesn't work) */
         unsigned has_tg4_compare : 1;
         /* color varyings should be flat shaded */
         unsigned flatshade : 1;
         unsigned has_128bit_rt : 1;
         unsigned use_xfb_emu : 1;
         unsigned rt_pack_rgba16 : PIPE_MAX_COLOR_BUFS;
      };
      uint32_t global;
   };

   int num_texture_states;
   nir_lower_tex_shadow_swizzle tex_swizzle[16];
   enum compare_func tex_compare_func[16];

   /* Compared depth32f samplers (D24 storage, so the hardware clamps the
    * border depth) with a border depth outside [0, 1]: the axes that wrap
    * to the border, and the border depth the shader uses for border texels.
    */
   unsigned tex_float_border_s : 16;
   unsigned tex_float_border_t : 16;
   float tex_border_depth[16];

   unsigned tex_is_128bit : 16;
   unsigned tex_mag_switchover : 16;
   /* samplers with unnormalized coordinates (OpenCL): the shader divides
    * the coordinates by the texture size */
   unsigned tex_unnormalized : 16;
   /* compute: 8-bit UNORM and SNORM textures, and the samplers that filter
    * nearest: those reads snap to the exact value (etna_launch_grid) */
   unsigned tex_unorm8 : 16;
   unsigned tex_snorm8 : 16;
   unsigned tex_nearest : 16;
   /* compute: images bound with a 128-bit format, for image access that
    * takes its format from the descriptor (OpenCL) */
   unsigned img_128bit : 16;
   /* compute: images (and buffer textures, 8 + slot) with a one-component
    * format, for image loads without a format in the shader (OpenCL) */
   unsigned img_1comp : 16;
   /* compute-only contexts (OpenCL): the 2D textures (0-7, or all in a wide
    * dispatch) that the shader reads with img_load where it can (images
    * 8 + texture, see etna_compute_swap_textures()), and of those, the ones
    * whose sampler takes the texel at the coordinates (nearest,
    * unnormalized, clamped to the edge or to the border), so that samples
    * are read that way too; of those, the ones clamped to the border, and
    * the ones whose border alpha is 1 (formats without alpha) */
   unsigned tex_img_read : 32;
   unsigned tex_img_sampled : 32;
   unsigned tex_img_border : 32;
   unsigned tex_img_border_a1 : 32;
   /* compute-only contexts (OpenCL): a dispatch of more than 16 textures
    * (etna_compute_probe_wide()), whose textures the texture unit reads take
    * the fragment slots in order (remap_tex_slots()); and the textures 8-31
    * it reads as images 16-39 that are 128-bit or have one component */
   unsigned cs_tex_wide : 1;
   unsigned tex_img_128bit : 32;
   unsigned tex_img_1comp : 32;
   /* compute: RGBA16 UNORM textures, sampled as integers and converted, and
    * the textures with a linear sampler that the shader filters
    * (lower_shader_linear) */
   unsigned tex_unorm16 : 16;
   unsigned tex_shader_linear : 16;
   /* compute-only contexts (OpenCL): array textures, whose layer the shader
    * rounds half to even (lower_layer_rint) */
   unsigned tex_layer_rint : 16;
   /* compute: the workgroup size of the dispatch, for a kernel of variable
    * workgroup size (OpenCL) with barriers that doesn't fit it unfolded: the
    * variant is compiled for that size, and folded (etna_compile_shader()) */
   uint16_t cs_block[3];
   unsigned sampler_companion[16];

   unsigned rt_is_128bit : ETNA_MAX_128BIT_RTS;
   unsigned rt_companion[ETNA_MAX_128BIT_RTS];
};

static inline bool
etna_shader_key_equal(const struct etna_shader_key* const a,
                      const struct etna_shader_key* const b)
{
   /* slow-path if we need to check tex_{swizzle,compare_func} */
   if (unlikely(a->has_sample_tex_compare || b->has_sample_tex_compare) ||
       unlikely(a->has_tg4_compare || b->has_tg4_compare) ||
       unlikely(a->tex_is_128bit || b->tex_is_128bit) ||
       unlikely(a->tex_mag_switchover || b->tex_mag_switchover) ||
       unlikely(a->tex_unnormalized || b->tex_unnormalized) ||
       unlikely(a->tex_unorm8 || b->tex_unorm8 || a->tex_snorm8 || b->tex_snorm8) ||
       unlikely(a->img_128bit || b->img_128bit || a->img_1comp || b->img_1comp) ||
       unlikely(a->tex_img_read || b->tex_img_read) ||
       unlikely(a->cs_tex_wide || b->cs_tex_wide) ||
       unlikely(a->tex_unorm16 || b->tex_unorm16) ||
       unlikely(a->tex_shader_linear || b->tex_shader_linear) ||
       unlikely(a->tex_layer_rint || b->tex_layer_rint) ||
       unlikely(a->cs_block[0] || b->cs_block[0]) ||
       unlikely(a->rt_is_128bit || b->rt_is_128bit))
      return memcmp(a, b, sizeof(struct etna_shader_key)) == 0;
   else
      return a->global == b->global;
}

struct etna_shader {
   /* shader id (for debug): */
   uint32_t id;
   uint32_t variant_count;

   struct nir_shader *nir;
   const struct etna_core_info *info;
   const struct etna_specs *specs;
   struct etna_compiler *compiler;
   struct etna_screen *screen;

   uint16_t tex_lod_samplers;
   uint16_t tg4_shadow_samplers;
   /* OpenCL pairs any image with any sampler; the hardware samples a texture
    * slot with that slot's sampler state. Texture slots in sampler_remap take
    * the state of sampler sampler_for_texture[] (see etna_launch_grid). */
   uint32_t sampler_remap;
   uint8_t sampler_for_texture[32];
   uint32_t textures_used;

   struct util_shader_variant_list variants;

   cache_key cache_key;     /* shader disk-cache key */

   /* parallel shader compiles */
   struct util_queue_fence ready;
};

bool
etna_shader_link(struct etna_context *ctx);

bool
etna_shader_upload_code(struct etna_context *ctx, struct etna_shader_variant *v);

bool
etna_shader_upload_constant_data(struct etna_context *ctx, struct etna_shader_variant *v);

bool
etna_shader_update_vertex(struct etna_context *ctx);

struct etna_shader_variant *
etna_shader_variant(struct etna_shader *shader,
                    const struct etna_shader_key* const key,
                    struct util_debug_callback *debug,
                    bool called_from_draw);

void
etna_shader_init(struct pipe_context *pctx);

bool
etna_shader_screen_init(struct pipe_screen *pscreen);

void
etna_shader_screen_fini(struct pipe_screen *pscreen);

#endif
