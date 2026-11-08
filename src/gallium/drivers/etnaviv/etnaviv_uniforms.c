/*
 * Copyright (c) 2016 Etnaviv Project
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

#include "etnaviv_uniforms.h"

#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_util.h"
#include "etnaviv_emit.h"
#include "pipe/p_defines.h"
#include "util/u_math.h"

static unsigned
get_const_idx(const struct etna_context *ctx, bool frag, unsigned samp_id)
{
   struct etna_screen *screen = ctx->screen;

   if (frag)
      return samp_id;

   return samp_id + screen->specs.vertex_sampler_offset;
}

static uint32_t
get_texrect_scale(const struct etna_context *ctx, bool frag,
                  enum etna_uniform_contents contents, uint32_t data)
{
   unsigned index = get_const_idx(ctx, frag, data);
   struct pipe_sampler_view *texture = ctx->sampler_view[index];
   uint32_t dim;

   if (contents == ETNA_UNIFORM_TEXRECT_SCALE_X)
      dim = texture->texture->width0;
   else
      dim = texture->texture->height0;

   return fui(1.0f / dim);
}

static inline bool
is_array_texture(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return true;
   default:
      return false;
   }
}

static uint32_t
get_texture_size(const struct etna_context *ctx, bool frag,
                  enum etna_uniform_contents contents, uint32_t data)
{
   unsigned index = get_const_idx(ctx, frag, data);
   struct pipe_sampler_view *texture = ctx->sampler_view[index];

   switch (contents) {
   case ETNA_UNIFORM_TEXTURE_WIDTH:
      if (texture->target == PIPE_BUFFER) {
         return texture->u.buf.size / util_format_get_blocksize(texture->format);
      } else {
         return u_minify(texture->texture->width0, texture->u.tex.first_level);
      }
   case ETNA_UNIFORM_TEXTURE_HEIGHT:
      return u_minify(texture->texture->height0, texture->u.tex.first_level);
   case ETNA_UNIFORM_TEXTURE_DEPTH:
      assert(texture->target != PIPE_BUFFER);

      if (is_array_texture(texture->target)) {
         if (texture->target != PIPE_TEXTURE_CUBE_ARRAY) {
            return texture->texture->array_size;
         } else {
            assert(texture->texture->array_size % 6 == 0);
            return texture->texture->array_size / 6;
         }
      }

      return u_minify(texture->texture->depth0, texture->u.tex.first_level);
   default:
      UNREACHABLE("Bad texture size field");
   }
}

static uint32_t
get_sampler_lod(const struct etna_context *ctx, bool frag,
                enum etna_uniform_contents contents, uint32_t data)
{
   unsigned index = get_const_idx(ctx, frag, data);
   const struct pipe_sampler_state *sampler = ctx->sampler[index];
   const struct pipe_sampler_view *view = ctx->sampler_view[index];
   const bool mipmap = sampler->min_mip_filter != PIPE_TEX_MIPFILTER_NONE;
   const float max_level = view->u.tex.last_level - view->u.tex.first_level;

   switch (contents) {
   case ETNA_UNIFORM_SAMPLER_LOD_MIN:
      return mipmap ? fui(CLAMP(sampler->min_lod, 0.0f, max_level)) : fui(0.0f);
   case ETNA_UNIFORM_SAMPLER_LOD_MAX:
      return mipmap ? fui(CLAMP(sampler->max_lod, 0.0f, max_level)) : fui(0.0f);
   case ETNA_UNIFORM_SAMPLER_LOD_BIAS:
      return fui(sampler->lod_bias);
   default:
      UNREACHABLE("Bad sampler lod field");
   }
}

/* Format word of an image descriptor, from NXP's driver for the formats it
 * was seen with (r32ui, r32f, rgba8, rgba8ui, rgba16f, rg32f, rg32ui), and
 * measured on GC7000 r6204:
 *   bits 0-2   log2(bytes per texel), for the address
 *   bits 4-5   2 for one 32-bit component, 1 otherwise (what fills register
 *              lanes the format does not have: 0 zero, 2 ones, 3 a copy)
 *   bits 6-9   type: 0 f32, 1 f16, 5 u32, 6 u16?, 7 u8, 2 s32?, 3 s16?,
 *              4 s8?, plus 8 for normalized (no effect seen on GC7000 r6204)
 *   bits 10-11 layout: 0 linear, 1 4x4 tiled, 2 super tiled (NXP's
 *              textures; not tested here)
 *   bit 12     set; bits 14-15 number of components (4 as 0)
 *   bits 16-31 swizzle: 4 bits per channel, 0-3 source, 4 zero, 5 one
 * 128-bit formats are two planes of 64-bit texels, each with its own
 * descriptor. Returns 0 for formats without a hardware image path. */
static uint32_t
etna_image_format_word(enum pipe_format format, enum etna_surface_layout layout)
{
   unsigned log2_bpp, cls, type, comps;

   switch (format) {
   case PIPE_FORMAT_R32G32B32A32_FLOAT: log2_bpp = 3; cls = 1; type = 0;  comps = 2; break;
   case PIPE_FORMAT_R32G32B32A32_UINT:  log2_bpp = 3; cls = 1; type = 5;  comps = 2; break;
   case PIPE_FORMAT_R32G32B32A32_SINT:  log2_bpp = 3; cls = 1; type = 2;  comps = 2; break;
   case PIPE_FORMAT_R16G16B16A16_FLOAT: log2_bpp = 3; cls = 1; type = 1;  comps = 4; break;
   case PIPE_FORMAT_R16G16B16A16_UINT:  log2_bpp = 3; cls = 1; type = 6;  comps = 4; break;
   case PIPE_FORMAT_R16G16B16A16_SINT:  log2_bpp = 3; cls = 1; type = 3;  comps = 4; break;
   case PIPE_FORMAT_R32_FLOAT:          log2_bpp = 2; cls = 2; type = 0;  comps = 1; break;
   case PIPE_FORMAT_R32_UINT:           log2_bpp = 2; cls = 2; type = 5;  comps = 1; break;
   case PIPE_FORMAT_R32_SINT:           log2_bpp = 2; cls = 2; type = 2;  comps = 1; break;
   case PIPE_FORMAT_R8G8B8A8_UNORM:     log2_bpp = 2; cls = 1; type = 15; comps = 4; break;
   case PIPE_FORMAT_R8G8B8A8_SNORM:     log2_bpp = 2; cls = 1; type = 12; comps = 4; break;
   case PIPE_FORMAT_R8G8B8A8_UINT:      log2_bpp = 2; cls = 1; type = 7;  comps = 4; break;
   case PIPE_FORMAT_R8G8B8A8_SINT:      log2_bpp = 2; cls = 1; type = 4;  comps = 4; break;
   default:
      return 0;
   }

   static const uint32_t swizzle[] = { 0, 0x5440, 0x5410, 0x5210, 0x3210 };

   const unsigned tiling = layout == ETNA_LAYOUT_TILED ? 1 : layout == ETNA_LAYOUT_SUPER_TILED ? 2 : 0;

   return log2_bpp | cls << 4 | type << 6 | tiling << 10 | 1 << 12 |
          (comps & 3) << 14 | swizzle[comps] << 16;
}

void
etna_uniforms_write(const struct etna_context *ctx,
                    const struct etna_shader_variant *sobj,
                    struct pipe_constant_buffer *cb)
{
   struct etna_screen *screen = ctx->screen;
   struct etna_cmd_stream *stream = ctx->stream;
   const struct etna_shader_uniform_info *uinfo = &sobj->uniforms;
   /* compute shaders run on the pixel shader unit, with PS_UNIFORM_BASE 0 */
   bool frag = (sobj == ctx->shader.fs) || sobj->stage == MESA_SHADER_COMPUTE;
   uint32_t base = frag ? screen->specs.ps_uniforms_offset : screen->specs.vs_uniforms_offset;

   if (screen->specs.has_unified_uniforms && sobj == ctx->shader.fs)
      base += ctx->shader.vs->uniforms.count * 4;

   if (!uinfo->count)
      return;

   etna_cmd_stream_reserve(stream, align(uinfo->count + 1, 2));
   etna_emit_load_state(stream, base >> 2, uinfo->count, 0);

   for (uint32_t i = 0; i < uinfo->count; i++) {
      uint32_t val = uinfo->data[i];

      switch (uinfo->contents[i]) {
      case ETNA_UNIFORM_CONSTANT:
         etna_cmd_stream_emit(stream, val);
         break;

      case ETNA_UNIFORM_UNIFORM:
         assert(cb->user_buffer && val * 4 < cb->buffer_size);
         etna_cmd_stream_emit(stream, ((uint32_t*) cb->user_buffer)[val]);
         break;

      case ETNA_UNIFORM_TEXRECT_SCALE_X:
      case ETNA_UNIFORM_TEXRECT_SCALE_Y:
         etna_cmd_stream_emit(stream,
            get_texrect_scale(ctx, frag, uinfo->contents[i], val));
         break;

      case ETNA_UNIFORM_TEXTURE_WIDTH:
      case ETNA_UNIFORM_TEXTURE_HEIGHT:
      case ETNA_UNIFORM_TEXTURE_DEPTH:
         etna_cmd_stream_emit(stream,
            get_texture_size(ctx, frag, uinfo->contents[i], val));
         break;

      case ETNA_UNIFORM_SAMPLER_LOD_MIN:
      case ETNA_UNIFORM_SAMPLER_LOD_MAX:
      case ETNA_UNIFORM_SAMPLER_LOD_BIAS:
         etna_cmd_stream_emit(stream,
            get_sampler_lod(ctx, frag, uinfo->contents[i], val));
         break;

      case ETNA_UNIFORM_UBO_ADDR:
         etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
            .bo = etna_buffer_resource(cb[val].buffer)->bo,
            .flags = ETNA_RELOC_READ,
            .offset = cb[val].buffer_offset,
         });
         break;

      case ETNA_UNIFORM_CONSTANT_DATA_ADDR:
         etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
            .bo = sobj->constant_bo,
            .flags = ETNA_RELOC_READ,
            .offset = val,
         });
         break;

      case ETNA_UNIFORM_XFB_ADDR:
         etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
            .bo = etna_buffer_resource(ctx->streamout.targets[val]->buffer)->bo,
            .flags = ETNA_RELOC_WRITE,
            .offset = ctx->streamout.targets[val]->buffer_offset +
                      ctx->streamout.captured_bytes[val],
         });
         break;

      case ETNA_UNIFORM_XFB_NUM_VERTICES:
         etna_cmd_stream_emit(stream, ctx->streamout.num_vertices);
         break;

      case ETNA_UNIFORM_XFB_FIRST_VERTEX:
         etna_cmd_stream_emit(stream, ctx->streamout.first_vertex);
         break;

      case ETNA_UNIFORM_SSBO_ADDR: {
         const struct pipe_shader_buffer *sb = &ctx->shader_buffers[sobj->stage].sb[val];

         if (sb->buffer) {
            etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
               .bo = etna_buffer_resource(sb->buffer)->bo,
               .flags = ETNA_RELOC_READ | ETNA_RELOC_WRITE,
               .offset = sb->buffer_offset,
            });
         } else {
            etna_cmd_stream_emit(stream, 0);
         }
      } break;

      case ETNA_UNIFORM_SSBO_SIZE:
         etna_cmd_stream_emit(stream, ctx->shader_buffers[sobj->stage].sb[val].buffer_size);
         break;

      case ETNA_UNIFORM_NUM_WORKGROUPS:
         etna_cmd_stream_emit(stream, ctx->compute_grid[val]);
         break;

      case ETNA_UNIFORM_IMAGE_ADDR: {
         const struct pipe_image_view *img = &ctx->shader_images[sobj->stage].img[val];
         const struct etna_resource *rsc = etna_resource(img->resource);
         const struct etna_resource_level *lev = &rsc->levels[img->u.tex.level];

         etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
            .bo = rsc->bo,
            .flags = ETNA_RELOC_READ | ETNA_RELOC_WRITE,
            .offset = lev->offset + img->u.tex.first_layer * lev->layer_stride,
         });
      } break;

      case ETNA_UNIFORM_IMAGE_STRIDE:
      case ETNA_UNIFORM_IMAGE_LAYER_STRIDE:
      case ETNA_UNIFORM_IMAGE_WIDTH:
      case ETNA_UNIFORM_IMAGE_HEIGHT:
      case ETNA_UNIFORM_IMAGE_DEPTH:
      case ETNA_UNIFORM_IMAGE_TILED:
      case ETNA_UNIFORM_IMAGE_PLANE_OFFSET: {
         const struct pipe_image_view *img = &ctx->shader_images[sobj->stage].img[val];
         const struct etna_resource *rsc = etna_resource(img->resource);
         const struct etna_resource_level *lev = &rsc->levels[img->u.tex.level];
         const uint32_t v[] = {
            lev->stride, lev->layer_stride, lev->width, lev->height,
            img->resource->target == PIPE_TEXTURE_3D ? lev->depth
               : img->u.tex.last_layer - img->u.tex.first_layer + 1,
            rsc->layout == ETNA_LAYOUT_TILED,
            etna_resource_level_second_plane_offset(lev),
         };

         etna_cmd_stream_emit(stream, v[uinfo->contents[i] - ETNA_UNIFORM_IMAGE_STRIDE]);
      } break;

      case ETNA_UNIFORM_IMAGE_DESC: {
         const unsigned index = val & 0xf, plane = (val >> 4) & 0xf, word = val >> 8;
         const struct pipe_image_view *img = &ctx->shader_images[sobj->stage].img[index];
         const struct etna_resource *rsc = etna_resource(img->resource);

         if (!rsc) {
            etna_cmd_stream_emit(stream, 0);
            break;
         }

         const struct etna_resource_level *lev = &rsc->levels[img->u.tex.level];

         if (word == 0) {
            etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
               .bo = rsc->bo,
               .flags = ETNA_RELOC_READ | ETNA_RELOC_WRITE,
               .offset = lev->offset + img->u.tex.first_layer * lev->layer_stride +
                         plane * etna_resource_level_second_plane_offset(lev),
            });
            break;
         }

         /* stride: of a row of texels, also for tiled images */
         const uint32_t v[4] = {
            0,
            lev->stride,
            lev->height << 16 | lev->width,
            etna_image_format_word(img->format, rsc->layout),
         };

         etna_cmd_stream_emit(stream, v[word & 3]);
      } break;

      case ETNA_UNIFORM_SHARED_ADDR:
         etna_cmd_stream_reloc(stream, &(struct etna_reloc) {
            .bo = ctx->compute_shared_bo,
            .flags = ETNA_RELOC_READ | ETNA_RELOC_WRITE,
         });
         break;

      case ETNA_UNIFORM_UNUSED:
         etna_cmd_stream_emit(stream, 0);
         break;
      }
   }

   if ((uinfo->count % 2) == 0)
      etna_cmd_stream_emit(stream, 0);
}

void
etna_set_shader_uniforms_dirty_flags(struct etna_shader_variant *sobj)
{
   uint32_t dirty = 0;
   uint32_t ubo_mask = 0;

   for (uint32_t i = 0; i < sobj->uniforms.count; i++) {
      switch (sobj->uniforms.contents[i]) {
      default:
         break;

      case ETNA_UNIFORM_UBO_ADDR:
         ubo_mask |= BITFIELD_BIT(sobj->uniforms.data[i]);
         break;

      case ETNA_UNIFORM_TEXRECT_SCALE_X:
      case ETNA_UNIFORM_TEXRECT_SCALE_Y:
      case ETNA_UNIFORM_TEXTURE_WIDTH:
      case ETNA_UNIFORM_TEXTURE_HEIGHT:
      case ETNA_UNIFORM_TEXTURE_DEPTH:
         dirty |= ETNA_DIRTY_SAMPLER_VIEWS;
         break;

      case ETNA_UNIFORM_SAMPLER_LOD_MIN:
      case ETNA_UNIFORM_SAMPLER_LOD_MAX:
      case ETNA_UNIFORM_SAMPLER_LOD_BIAS:
         dirty |= ETNA_DIRTY_SAMPLERS | ETNA_DIRTY_SAMPLER_VIEWS;
         break;

      case ETNA_UNIFORM_XFB_ADDR:
      case ETNA_UNIFORM_XFB_NUM_VERTICES:
      case ETNA_UNIFORM_XFB_FIRST_VERTEX:
         dirty |= ETNA_DIRTY_STREAMOUT;
         break;
      }
   }

   sobj->uniforms_dirty_bits = dirty;
   sobj->ubo_mask = ubo_mask;
}
