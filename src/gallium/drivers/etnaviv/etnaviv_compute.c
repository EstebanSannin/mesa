/*
 * Copyright (c) 2026 Etnaviv Project
 * SPDX-License-Identifier: MIT
 *
 * Experimental compute support (ETNA_EXPERIMENTAL_COMPUTE=1).
 *
 * Compute shaders run on the pixel shader unit. The work-item IDs are written
 * by the thread walker into the first temporaries: t0 = global invocation ID,
 * t1 = local invocation ID, t2 = workgroup ID (CL_CONFIG value order 3).
 *
 * Based on the dispatch sequence from the experimental compute support by
 * Italo Nicola and Tomeu Vizoso (Mesa MR !18986) and the kernel PPU flop reset
 * (drivers/gpu/drm/etnaviv/etnaviv_flop_reset.c).
 */

#include "etnaviv_clear_blit.h"
#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_emit.h"
#include "etnaviv_format.h"
#include "etnaviv_resource.h"
#include "etnaviv_screen.h"
#include "etnaviv_shader.h"
#include "etnaviv_state.h"
#include "etnaviv_texture.h"
#include "etnaviv_uniforms.h"

#include "hw/common.xml.h"
#include "hw/state.xml.h"
#include "hw/state_3d.xml.h"

#include "util/u_debug.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"

static void
etna_set_shader_buffers(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start, unsigned count,
                        const struct pipe_shader_buffer *buffers,
                        unsigned writable_bitmask)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_shader_buffer_state *so = &ctx->shader_buffers[shader];

   assert(start + count <= ETNA_MAX_SHADER_BUFFERS);

   for (unsigned i = 0; i < count; i++) {
      struct pipe_shader_buffer *dst = &so->sb[start + i];

      if (buffers && buffers[i].buffer) {
         pipe_resource_reference(&dst->buffer, buffers[i].buffer);
         dst->buffer_offset = buffers[i].buffer_offset;
         dst->buffer_size = buffers[i].buffer_size;
         so->enabled_mask |= BITFIELD_BIT(start + i);
      } else {
         pipe_resource_reference(&dst->buffer, NULL);
         so->enabled_mask &= ~BITFIELD_BIT(start + i);
      }
   }
}

static void
etna_set_shader_images(struct pipe_context *pctx, mesa_shader_stage shader,
                       unsigned start, unsigned count, unsigned unbind_num_trailing_slots,
                       const struct pipe_image_view *images)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_shader_image_state *so = &ctx->shader_images[shader];

   for (unsigned i = 0; i < count + unbind_num_trailing_slots; i++) {
      const unsigned n = start + i;
      struct pipe_image_view *dst = &so->img[n];

      if (n >= ETNA_MAX_SHADER_IMAGES)
         break;

      /* buffers too: OpenCL 1D image buffers */
      if (i < count && images && images[i].resource) {
         util_copy_image_view(dst, &images[i]);
         so->enabled_mask |= BITFIELD_BIT(n);
      } else {
         util_copy_image_view(dst, NULL);
         so->enabled_mask &= ~BITFIELD_BIT(n);
      }
   }
}

/* Bring the linear base of an image up to date: newer contents may sit in
 * the render sibling or behind tile status. */
static void
etna_compute_prepare_image(struct etna_context *ctx, const struct pipe_image_view *img)
{
   if (img->resource->target == PIPE_BUFFER)
      return;

   struct etna_resource *base = etna_resource(img->resource);
   const unsigned lvl = img->u.tex.level;

   if (base->render && etna_resource_newer(etna_resource(base->render), base))
      etna_copy_resource(&ctx->base, &base->base, base->render, lvl, lvl, false);
   else if (etna_resource_needs_flush(base))
      etna_copy_resource(&ctx->base, &base->base, &base->base, lvl, lvl, false);
}

static void *
etna_create_compute_state(struct pipe_context *pctx,
                          const struct pipe_compute_state *pcs)
{
   if (pcs->ir_type != PIPE_SHADER_IR_NIR)
      return NULL;

   struct pipe_shader_state pss = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = (struct nir_shader *)pcs->prog,
   };

   /* OpenCL kernels (rusticl) arrive lowered to explicit memory access and
    * kernel arguments in constant buffer 0: run them as compute shaders */
   if (pss.ir.nir->info.stage == MESA_SHADER_KERNEL)
      pss.ir.nir->info.stage = MESA_SHADER_COMPUTE;

   return pctx->create_fs_state(pctx, &pss);
}

static void
etna_bind_compute_state(struct pipe_context *pctx, void *cso)
{
   struct etna_context *ctx = etna_context(pctx);

   ctx->shader.bind_cs = cso;
   ctx->shader.cs = NULL;
}

static void
etna_delete_compute_state(struct pipe_context *pctx, void *cso)
{
   pctx->delete_fs_state(pctx, cso);
}

/* does a workgroup of n invocations fit with barrier(): invocations (per
 * thread of a folded variant) x temporaries (in pairs) within
 * etna_cs_barrier_capacity(), as etna_launch_grid() checks */
static bool
etna_cs_workgroup_fits(struct etna_context *ctx, const struct etna_shader_variant *v, unsigned n)
{
   const unsigned temps = align(MAX3(v->num_temps, MAX2(v->cs_input_count, 1),
                                     ctx->compute_min_temps), 2);

   return DIV_ROUND_UP(n, MAX2(v->cs_fold, 1)) * temps <= etna_cs_barrier_capacity(ctx->screen->info);
}

static void
etna_get_compute_state_info(struct pipe_context *pctx, void *cso,
                            struct pipe_compute_state_object_info *info)
{
   struct etna_screen *screen = etna_screen(pctx->screen);
   struct etna_shader_key key = {};
   struct etna_shader_variant *v = etna_shader_variant(cso, &key, &pctx->debug, true);

   info->max_threads = screen->base.compute_caps.max_threads_per_block;

   /* With barrier() a whole workgroup must be resident: invocations x
    * temporaries (in pairs) within etna_cs_barrier_capacity(), as
    * etna_launch_grid() checks. Report the largest multiple of 4 invocations
    * that fits. */
   if (v && v->cs_uses_barrier && v->cs_fold <= 1) {
      const unsigned temps = align(MAX3(v->num_temps, MAX2(v->cs_input_count, 1),
                                        etna_context(pctx)->compute_min_temps), 2);
      const unsigned fit = etna_cs_barrier_capacity(screen->info) / temps;

      info->max_threads = MIN2(info->max_threads, MAX2(fit & ~3u, 4));

      /* A kernel of variable workgroup size (OpenCL) gets a folded variant
       * for a workgroup that doesn't fit (etna_launch_grid()): offer the
       * largest power of two whose folded variant fits, as a row. A smaller
       * workgroup takes a smaller fold factor, which needn't divide it. */
      struct etna_shader *shader = cso;
      if (shader->nir->info.workgroup_size_variable) {
         for (unsigned n = screen->base.compute_caps.max_threads_per_block;
              n > info->max_threads; n /= 2) {
            struct etna_shader_key fkey = { .cs_block = { n, 1, 1 } };
            struct etna_shader_variant *fv =
               etna_shader_variant(cso, &fkey, &pctx->debug, true);

            if (fv && fv->cs_fold > 1 && etna_cs_workgroup_fits(etna_context(pctx), fv, n)) {
               info->max_threads = n;
               break;
            }
         }
      }
   }

   info->private_memory = v ? v->cs_private_size : 0;
   info->preferred_simd_size = 4;
   info->simd_sizes = 4;
}

static void
etna_memory_barrier(struct pipe_context *pctx, unsigned flags)
{
   struct etna_context *ctx = etna_context(pctx);

   etna_set_state(ctx->stream, VIVS_GL_FLUSH_CACHE,
                  VIVS_GL_FLUSH_CACHE_TEXTURE | VIVS_GL_FLUSH_CACHE_SHADER_L1 |
                  VIVS_GL_FLUSH_CACHE_UNK10 | VIVS_GL_FLUSH_CACHE_UNK11);
   etna_stall(ctx->stream, SYNC_RECIPIENT_FE, SYNC_RECIPIENT_PE);
}

/* Compute and fragment shaders sample through the same slots. Bind the
 * compute textures there for the dispatch (saving the fragment bindings in
 * *saved), or put the fragment bindings back. */
struct etna_saved_textures {
   void *sampler[PIPE_MAX_SAMPLERS];
   struct pipe_sampler_view *view[PIPE_MAX_SAMPLERS];
   unsigned num_samplers, num_views;
};

/* A texture slot only works with a sampler state, but OpenCL reads images
 * without one (texel fetch): such textures get a nearest, clamp-to-edge one. */
static void *
etna_compute_fetch_sampler(struct etna_context *ctx)
{
   if (!ctx->cs_fetch_sampler) {
      const struct pipe_sampler_state ss = {
         .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .min_img_filter = PIPE_TEX_FILTER_NEAREST,
         .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
         .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
      };

      ctx->cs_fetch_sampler = ctx->base.create_sampler_state(&ctx->base, &ss);
   }

   return ctx->cs_fetch_sampler;
}

/* An OpenCL 2D texture that the image instructions can read: a format of up
 * to 64 bits with an image format word, linear or 4x4 tiled, and no swizzle
 * other than the format's own (the descriptor gives the format's channels,
 * a missing one as 0, and alpha as 1). */
static bool
etna_compute_view_reads_as_image(const struct pipe_sampler_view *sv)
{
   const struct etna_resource *rsc = etna_resource(sv->texture);
   const unsigned comps = util_format_get_nr_components(sv->format);
   const unsigned char swizzle[4] = { sv->swizzle_r, sv->swizzle_g, sv->swizzle_b, sv->swizzle_a };

   if (sv->target != PIPE_TEXTURE_2D || !etna_format_is_image_format(sv->format) ||
       util_format_get_blocksize(sv->format) > 8 ||
       (rsc->layout != ETNA_LAYOUT_LINEAR && rsc->layout != ETNA_LAYOUT_TILED))
      return false;

   for (unsigned c = 0; c < 4; c++) {
      if (swizzle[c] != PIPE_SWIZZLE_X + c &&
          !(c >= comps && swizzle[c] == (c == 3 ? PIPE_SWIZZLE_1 : PIPE_SWIZZLE_0)))
         return false;
   }

   return true;
}

static void
etna_compute_swap_textures(struct etna_context *ctx, struct etna_saved_textures *saved,
                           bool compute, const struct etna_shader *shader)
{
   struct pipe_context *pctx = &ctx->base;
   const unsigned n = PIPE_MAX_SAMPLERS / 2;
   void *samplers[PIPE_MAX_SAMPLERS] = { NULL };
   struct pipe_sampler_view *views[PIPE_MAX_SAMPLERS] = { NULL };

   if (compute) {
      saved->num_samplers = ctx->num_fragment_samplers;
      saved->num_views = ctx->num_fragment_sampler_views;
      for (unsigned i = 0; i < n; i++) {
         saved->sampler[i] = ctx->sampler[i];
         saved->view[i] = NULL;
         pipe_sampler_view_reference(&saved->view[i], ctx->sampler_view[i]);
      }
      memcpy(samplers, ctx->cs_sampler, sizeof(void *) * n);
      memcpy(views, ctx->cs_sampler_view, sizeof(views[0]) * n);
      u_foreach_bit(t, shader->sampler_remap)
         samplers[t] = ctx->cs_sampler[shader->sampler_for_texture[t]];
      u_foreach_bit(t, shader->textures_used & BITFIELD_MASK(n)) {
         if (!samplers[t])
            samplers[t] = etna_compute_fetch_sampler(ctx);
      }

      /* buffers (OpenCL 1D image buffers) are read with the image
       * instructions as images 8 + slot, see lower_texbuf_to_image(): the
       * texture unit doesn't get them */
      ctx->cs_texbuf_mask = 0;
      for (unsigned t = 0; t < n; t++) {
         if (views[t] && views[t]->target == PIPE_BUFFER) {
            ctx->cs_texbuf_mask |= BITFIELD_BIT(t);
            views[t] = NULL;
         }
      }

      /* OpenCL 2D textures of formats up to 64 bits are read with img_load
       * as well, as NXP's OpenCL does. GC7000 r6204's texture unit fetches
       * about 165 M texels a second (95 M RGBA8), from its cache as from
       * memory; img_load reads RGBA8 twice as fast. 128-bit formats stay with
       * the texture unit: their two planes take two img_loads, which is
       * slower. The views stay bound for the reads that keep the texture unit
       * (lower_texbuf_to_image()). */
      ctx->cs_teximg_mask = 0;
      for (unsigned t = 0; t < MIN2(n, 8); t++) {
         if (ctx->compute_only && views[t] && etna_compute_view_reads_as_image(views[t])) {
            ctx->cs_teximg_mask |= BITFIELD_BIT(t);
            ctx->cs_texbuf_mask |= BITFIELD_BIT(t);
         }
      }

      /* Per slot, a copy of the sampler made for its view:
       * - the shader filters linear samplers (lower_shader_linear), bound as a
       *   nearest copy: RGBA16 UNORM samples as integers, which don't filter,
       *   and OpenCL (compute-only contexts) needs more precise weights than
       *   the texture unit's (an 8-bit UNORM sample at a weight of 0.6 was 1.6
       *   steps off, the CTS allows half a step);
       * - OpenCL's CL_ADDRESS_CLAMP border is (0, 0, 0, 1) for a format without
       *   alpha: the texture unit doesn't swizzle the border color. */
      ctx->cs_shader_linear = 0;
      for (unsigned t = 0; t < n; t++) {
         const struct pipe_sampler_state *ss = samplers[t];

         if (!views[t] || !ss)
            continue;

         struct pipe_sampler_state copy = *ss;
         bool changed = false;

         if ((views[t]->format == PIPE_FORMAT_R16G16B16A16_UNORM || ctx->compute_only) &&
             (ss->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
              ss->mag_img_filter != PIPE_TEX_FILTER_NEAREST)) {
            copy.min_img_filter = PIPE_TEX_FILTER_NEAREST;
            copy.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
            copy.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
            ctx->cs_shader_linear |= BITFIELD_BIT(t);
            changed = true;
         }

         if (ctx->compute_only && !util_format_has_alpha(views[t]->format) &&
             (ss->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
              ss->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
              ss->wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER)) {
            if (util_format_is_pure_integer(views[t]->format))
               copy.border_color.ui[3] = 1;
            else
               copy.border_color.f[3] = 1.0f;
            changed = true;
         }

         if (!changed)
            continue;

         if (!ctx->cs_sampler_copy[t] ||
             memcmp(&copy, &ctx->cs_sampler_copy_state[t], sizeof(copy))) {
            if (ctx->cs_sampler_copy[t])
               pctx->delete_sampler_state(pctx, ctx->cs_sampler_copy[t]);
            ctx->cs_sampler_copy[t] = pctx->create_sampler_state(pctx, &copy);
            ctx->cs_sampler_copy_state[t] = copy;
         }

         samplers[t] = ctx->cs_sampler_copy[t];
      }
   } else {
      memcpy(samplers, saved->sampler, sizeof(void *) * n);
      memcpy(views, saved->view, sizeof(views[0]) * n);
   }

   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, n, samplers);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0,
                           compute ? ctx->cs_num_sampler_views : saved->num_views, 0, views);
   if (!compute) {
      ctx->num_fragment_samplers = saved->num_samplers;
      for (unsigned i = 0; i < n; i++)
         pipe_sampler_view_reference(&saved->view[i], NULL);
   }
}

static bool
etna_get_grid(struct etna_context *ctx, const struct pipe_grid_info *info,
              uint32_t grid[3])
{
   if (info->indirect) {
      /* No indirect dispatch command is known: read the group counts back. */
      pipe_buffer_read(&ctx->base, info->indirect, info->indirect_offset,
                       3 * sizeof(uint32_t), grid);
   } else {
      memcpy(grid, info->grid, 3 * sizeof(uint32_t));
   }

   return grid[0] && grid[1] && grid[2];
}

/* The kernel's hang check kills a job when the GPU's front-end hasn't moved
 * for a period of 500 ms (the first check of a job usually extends it once).
 * A dispatch keeps the front-end waiting until it is done, so a dispatch of
 * more than about a second is killed, with its results incomplete. The
 * front-end moves on after each dispatch (the FE->PE stall in
 * etna_launch_grid()), so a long grid goes out in parts of an estimated
 * ETNA_CS_PART_MS each: whole Z slices, Y rows, or runs of X.
 * The estimate: one instruction per cycle per shader core at ETNA_CS_MHZ,
 * every instruction of the shader once per thread, 64 times when the shader
 * has loops. ETNA_CS_PART sets the workgroups per part, for testing. */
#define ETNA_CS_PART_MS 50
#define ETNA_CS_MHZ 800
/* at most this many parts, so that a dispatch fits into a command stream */
#define ETNA_CS_MAX_PARTS 256
/* command stream space a part takes at most: base workgroup uniforms, offset,
 * counts, kick, flushes and stall */
#define ETNA_CS_PART_DWORDS 48

static unsigned
etna_compute_parts(const struct etna_context *ctx, const struct etna_shader_variant *v,
                   const struct pipe_grid_info *info, const uint32_t grid[3], uint32_t part[3])
{
   const unsigned fold = MAX2(v->cs_fold, 1);
   const uint64_t threads = DIV_ROUND_UP(info->block[0] * info->block[1] * info->block[2], fold);
   const uint64_t group_cost = threads * (v->code_size / 4) * (v->cs_has_loops ? 64 : 1);
   const uint64_t budget = (uint64_t)ETNA_CS_PART_MS * ETNA_CS_MHZ * 1000 *
                           ctx->screen->info->gpu.shader_core_count;
   const uint64_t row = grid[0], slice = (uint64_t)grid[0] * grid[1];
   uint64_t max = debug_get_num_option("ETNA_CS_PART", 0);
   unsigned parts;

   if (!max)
      max = MAX2(budget / MAX2(group_cost, 1), 1);

   for (;;) {
      if (max >= slice) {
         part[0] = grid[0];
         part[1] = grid[1];
         part[2] = MIN2(grid[2], max / slice);
      } else if (max >= row) {
         part[0] = grid[0];
         part[1] = max / row;
         part[2] = 1;
      } else {
         part[0] = max;
         part[1] = 1;
         part[2] = 1;
      }

      parts = DIV_ROUND_UP(grid[0], part[0]) * DIV_ROUND_UP(grid[1], part[1]) *
              DIV_ROUND_UP(grid[2], part[2]);
      if (parts <= ETNA_CS_MAX_PARTS)
         return parts;
      max *= 2;
   }
}

static void
etna_launch_grid(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_cmd_stream *stream = ctx->stream;
   struct etna_shader *shader = ctx->shader.bind_cs;
   uint32_t grid[3];

   if (!shader || !etna_get_grid(ctx, info, grid))
      return;

   /* Compute shaders sample through the fragment slots. Bind the compute
    * textures there first: the shader variant depends on them, as for
    * fragment shaders (128-bit formats, shadow compare, ...). */
   struct etna_saved_textures saved;
   const uint32_t textures = shader->textures_used;
   struct etna_shader_key key = {};

   /* set by etna_compute_swap_textures() for this dispatch */
   ctx->cs_texbuf_mask = 0;
   ctx->cs_teximg_mask = 0;
   ctx->cs_shader_linear = 0;

   if (textures) {
      etna_compute_swap_textures(ctx, &saved, true, shader);
      etna_shader_key_set_fs_textures(ctx, &key, shader->tex_lod_samplers,
                                      shader->tg4_shadow_samplers);

      /* The texture unit converts 8-bit normalized texels to float slightly
       * off (UNORM 67 reads as 67.02 / 255, 128 as 128.06 / 255; 0 and 255
       * are exact), in 2D and 3D, sampled or fetched. OpenCL wants the exact
       * value: reads that don't filter snap back to it in the shader. */
      u_foreach_bit(i, textures & BITFIELD_MASK(16)) {
         const struct pipe_sampler_view *sv = ctx->sampler_view[i];
         const struct pipe_sampler_state *ss = ctx->sampler[i];

         if (!sv)
            continue;

         /* filters nearest (not a nearest copy of a linear sampler) */
         if (ss && ss->min_img_filter == PIPE_TEX_FILTER_NEAREST &&
             ss->mag_img_filter == PIPE_TEX_FILTER_NEAREST &&
             !(ctx->cs_shader_linear & BITFIELD_BIT(i)))
            key.tex_nearest |= BITFIELD_BIT(i);
         if (ctx->cs_shader_linear & BITFIELD_BIT(i))
            key.tex_shader_linear |= BITFIELD_BIT(i);

         /* OpenCL rounds the array layer half to even */
         if (ctx->compute_only && (sv->target == PIPE_TEXTURE_1D_ARRAY ||
                                   sv->target == PIPE_TEXTURE_2D_ARRAY))
            key.tex_layer_rint |= BITFIELD_BIT(i);

         const struct util_format_description *desc = util_format_description(sv->format);
         if (desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB)
            continue;

         if (util_format_is_unorm8(desc))
            key.tex_unorm8 |= BITFIELD_BIT(i);
         else if (util_format_is_snorm8(sv->format))
            key.tex_snorm8 |= BITFIELD_BIT(i);

         if (sv->format == PIPE_FORMAT_R16G16B16A16_UNORM)
            key.tex_unorm16 |= BITFIELD_BIT(i);
      }

      /* textures read with img_load, and the samplers whose samples can be */
      key.tex_img_read = ctx->cs_teximg_mask;
      u_foreach_bit(t, ctx->cs_teximg_mask) {
         const struct pipe_sampler_state *ss = ctx->sampler[t];

         if (ss && ss->unnormalized_coords && !(ctx->cs_shader_linear & BITFIELD_BIT(t)) &&
             ss->min_img_filter == PIPE_TEX_FILTER_NEAREST &&
             ss->mag_img_filter == PIPE_TEX_FILTER_NEAREST &&
             ss->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_EDGE && ss->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_EDGE)
            key.tex_img_sampled |= BITFIELD_BIT(t);
      }
   }

   /* 128-bit images are two planes: access without a format in the shader
    * (OpenCL) needs to know which */
   const struct etna_shader_image_state *bound = &ctx->shader_images[MESA_SHADER_COMPUTE];
   u_foreach_bit(i, bound->enabled_mask) {
      if (util_format_get_blocksize(bound->img[i].format) == 16)
         key.img_128bit |= BITFIELD_BIT(i);
      if (util_format_get_nr_components(bound->img[i].format) == 1)
         key.img_1comp |= BITFIELD_BIT(i);
   }
   u_foreach_bit(t, ctx->cs_texbuf_mask) {
      if (util_format_get_blocksize(ctx->cs_sampler_view[t]->format) == 16)
         key.img_128bit |= BITFIELD_BIT(ETNA_MAX_SHADER_IMAGES + t);
      if (util_format_get_nr_components(ctx->cs_sampler_view[t]->format) == 1)
         key.img_1comp |= BITFIELD_BIT(ETNA_MAX_SHADER_IMAGES + t);
   }

   struct etna_shader_variant *v =
      etna_shader_variant(shader, &key, &ctx->base.debug, true);
   if (!v)
      goto out;

   /* a kernel of variable workgroup size with barriers that doesn't fit this
    * workgroup unfolded: a variant compiled for it, folded */
   if (v->cs_uses_barrier && v->cs_fold <= 1 && shader->nir->info.workgroup_size_variable &&
       !etna_cs_workgroup_fits(ctx, v, info->block[0] * info->block[1] * info->block[2])) {
      for (unsigned i = 0; i < 3; i++)
         key.cs_block[i] = info->block[i];
      v = etna_shader_variant(shader, &key, &ctx->base.debug, true);
      if (!v)
         goto out;
   }

   ctx->shader.cs = v;

   uint32_t part[3];
   const unsigned parts = etna_compute_parts(ctx, v, info, grid, part);

   /* A forced flush of a full command stream must not cut the dispatch: make
    * room for it now, before any of its state and resources. The stream holds
    * 0x4000 dwords at most. */
   if (stream->offset + 1024 + v->uniforms.count + parts * ETNA_CS_PART_DWORDS > 0x4000 - 2)
      etna_cmd_stream_force_flush(stream);

   if (!etna_shader_upload_code(ctx, v) ||
       !etna_shader_upload_constant_data(ctx, v)) {
      BUG("failed to upload compute shader");
      goto out;
   }

   memcpy(ctx->compute_grid, grid, sizeof(ctx->compute_grid));
   memcpy(ctx->compute_block, info->block, sizeof(ctx->compute_block));

   /* one shared memory slice per workgroup of this dispatch: the shader's own
    * shared memory, then the __local kernel arguments (OpenCL) */
   ctx->compute_shared_slice = align(v->cs_shared_size + info->variable_shared_mem, 16);
   if (ctx->compute_shared_slice) {
      const uint64_t need = (uint64_t)ctx->compute_shared_slice * grid[0] * grid[1] * grid[2];

      if (need > UINT32_MAX) {
         BUG("shared memory for %u workgroups does not fit", grid[0] * grid[1] * grid[2]);
         goto out;
      }

      if (need > ctx->compute_shared_bo_size) {
         if (ctx->compute_shared_bo)
            etna_bo_del(ctx->compute_shared_bo);
         ctx->compute_shared_bo_size = align(need, 64 * 1024);
         ctx->compute_shared_bo = etna_bo_new(ctx->screen->dev, ctx->compute_shared_bo_size,
                                              DRM_ETNA_GEM_CACHE_WC);
         if (!ctx->compute_shared_bo) {
            ctx->compute_shared_bo_size = 0;
            BUG("no memory for compute shared memory");
            goto out;
         }
      }
   }

   /* scratch for the values that folded invocations keep across barrier():
    * 16 bytes per slot and invocation of the dispatch */
   if (v->cs_scratch_slots) {
      const uint64_t need = (uint64_t)v->cs_scratch_slots * 16 * grid[0] * grid[1] * grid[2] *
                            info->block[0] * info->block[1] * info->block[2];

      if (need > UINT32_MAX) {
         BUG("compute scratch for %u workgroups does not fit", grid[0] * grid[1] * grid[2]);
         goto out;
      }

      if (need > ctx->compute_scratch_bo_size) {
         if (ctx->compute_scratch_bo)
            etna_bo_del(ctx->compute_scratch_bo);
         ctx->compute_scratch_bo_size = align(need, 64 * 1024);
         ctx->compute_scratch_bo = etna_bo_new(ctx->screen->dev, ctx->compute_scratch_bo_size,
                                               DRM_ETNA_GEM_CACHE_WC);
         if (!ctx->compute_scratch_bo) {
            ctx->compute_scratch_bo_size = 0;
            BUG("no memory for compute scratch");
            goto out;
         }
      }
   }

   /* OpenCL private memory: one slot per invocation of the dispatch */
   if (v->cs_private_size) {
      const uint64_t need = (uint64_t)v->cs_private_size * grid[0] * grid[1] * grid[2] *
                            info->block[0] * info->block[1] * info->block[2];

      if (need > 512 * 1024 * 1024) {
         BUG("private memory of %" PRIu64 " bytes for the dispatch is too large, dispatch skipped", need);
         goto out;
      }

      if (need > ctx->compute_private_bo_size) {
         if (ctx->compute_private_bo)
            etna_bo_del(ctx->compute_private_bo);
         ctx->compute_private_bo_size = align(need, 64 * 1024);
         ctx->compute_private_bo = etna_bo_new(ctx->screen->dev, ctx->compute_private_bo_size,
                                               DRM_ETNA_GEM_CACHE_WC);
         if (!ctx->compute_private_bo) {
            ctx->compute_private_bo_size = 0;
            BUG("no memory for private memory");
            goto out;
         }
      }
   }

   /* Make sure the frontend-visible state of the buffers is tracked. */
   struct etna_shader_buffer_state *sb = &ctx->shader_buffers[MESA_SHADER_COMPUTE];
   u_foreach_bit(i, sb->enabled_mask)
      resource_written(ctx, sb->sb[i].buffer);

   /* indirectly indexed uniforms are read from constant buffer 0 as a UBO */
   etna_upload_constbufs(ctx, MESA_SHADER_COMPUTE, v);

   struct etna_constbuf_state *cb = &ctx->constant_buffer[MESA_SHADER_COMPUTE];
   u_foreach_bit(i, cb->enabled_mask) {
      if (cb->cb[i].buffer)
         resource_read(ctx, cb->cb[i].buffer);
   }

   const unsigned input_count = MAX2(v->cs_input_count, 1);
   /* a folded variant runs the workgroup as a row of ceil(block / fold)
    * threads */
   const unsigned fold = MAX2(v->cs_fold, 1);
   const unsigned hw_block[3] = {
      fold > 1 ? DIV_ROUND_UP(info->block[0] * info->block[1] * info->block[2], fold) : info->block[0],
      fold > 1 ? 1 : info->block[1],
      fold > 1 ? 1 : info->block[2],
   };
   const unsigned block_size = hw_block[0] * hw_block[1] * hw_block[2];
   /* Threads of a workgroup that are resident at the same time, in units of
    * 4-wide threads per shader core. Too few and barrier() does not wait for the
    * whole workgroup; found by experiment on GC7000 r6204. */
   const unsigned thread_alloc = ctx->compute_thread_alloc ? ctx->compute_thread_alloc :
      DIV_ROUND_UP(block_size, 4 * ctx->screen->info->gpu.shader_core_count);
   const unsigned num_temps = MAX3(v->num_temps, input_count, ctx->compute_min_temps);
   const unsigned num_inst = v->code_size / 4;

   /* With barrier() the whole workgroup must be resident at once, and the
    * shader cores hold 512 invocation-temporaries each: invocations x temps
    * <= 1024 on GC7000 r6204 (2 cores), found by experiment at 32 to 256
    * invocations, but the boundary itself hangs under load, with the
    * temporaries counted in pairs, and 1008 hangs for some kernels: the
    * limit is etna_cs_barrier_capacity(). Beyond it barrier() never returns
    * and the GPU hangs for every client.
    * etna_compile_shader() folds such workgroups onto fewer threads; if even
    * that does not fit, skip the dispatch. */
   if (v->cs_uses_barrier &&
       block_size * align(num_temps, 2) > etna_cs_barrier_capacity(ctx->screen->info)) {
      BUG("compute: %u invocations x %u temporaries do not fit with barrier(), dispatch skipped",
          block_size, num_temps);
      goto out;
   }

   /* a dispatch can be the first job of a stream: set up the GPU like a draw */
   if (ctx->needs_gpu_state_reset)
      etna_reset_gpu_state(ctx);

   /* textures first: bringing a sampler source up to date may blit */
   struct etna_shader_image_state *images = &ctx->shader_images[MESA_SHADER_COMPUTE];
   u_foreach_bit(i, images->enabled_mask) {
      etna_compute_prepare_image(ctx, &images->img[i]);
      if (images->img[i].access & PIPE_IMAGE_ACCESS_WRITE)
         resource_written(ctx, images->img[i].resource);
      else
         resource_read(ctx, images->img[i].resource);
   }

   /* textures read with img_load: up to date in memory (no render copy or
    * tile status left), as the shader images */
   u_foreach_bit(t, ctx->cs_teximg_mask) {
      const struct pipe_sampler_view *sv = ctx->cs_sampler_view[t];
      const struct pipe_image_view img = {
         .resource = sv->texture,
         .format = sv->format,
         .access = PIPE_IMAGE_ACCESS_READ,
         .u.tex.first_layer = sv->u.tex.first_layer,
         .u.tex.last_layer = sv->u.tex.last_layer,
         .u.tex.level = sv->u.tex.first_level,
      };

      etna_compute_prepare_image(ctx, &img);
   }

   u_foreach_bit(t, ctx->cs_texbuf_mask)
      resource_read(ctx, ctx->cs_sampler_view[t]->texture);

   if (textures) {
      u_foreach_bit(i, ctx->active_sampler_views & textures) {
         etna_update_sampler_source(ctx->sampler_view[i], i);
         resource_read(ctx, ctx->sampler_view[i]->texture);
      }
      etna_set_state(stream, VIVS_GL_FLUSH_CACHE, VIVS_GL_FLUSH_CACHE_TEXTURE);
      ctx->emit_texture_state(ctx);
   }

   /* OpenGL, not OpenCL, API mode, like NXP's driver for GLES compute: in
    * OpenCL mode the texture unit takes the layer of a 2D array texture as
    * a coordinate normalized to the layer count */
   etna_set_state(stream, VIVS_GL_API_MODE, VIVS_GL_API_MODE_OPENGL);
   /* memory the CPU wrote (buffers, uploaded constants) may be stale in L1 */
   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, VIVS_GL_FLUSH_CACHE_SHADER_L1);
   etna_stall(stream, SYNC_RECIPIENT_FE, SYNC_RECIPIENT_PE);

   /* global buffers are only addressed from the kernel arguments: reference
    * them in the submit */
   u_foreach_bit(i, ctx->global_bindings_mask) {
      resource_written(ctx, ctx->global_bindings[i]);
      etna_cmd_stream_ref_bo(stream, etna_buffer_resource(ctx->global_bindings[i])->bo,
                             ETNA_RELOC_READ | ETNA_RELOC_WRITE);
   }

   etna_uniforms_write(ctx, v, ctx->constant_buffer[MESA_SHADER_COMPUTE].cb);

   etna_set_state(stream, VIVS_PS_VARYING_NUM_COMPONENTS(0), 0);
   etna_set_state(stream, VIVS_PA_VARYING_NUM_COMPONENTS(0), 0);
   etna_set_state(stream, VIVS_PS_SAMPLER_BASE, 0);
   etna_set_state(stream, VIVS_PS_UNIFORM_BASE, 0);
   /* HALTI5 takes the end of the PS range from PS_HALTI5_RANGE_HIGH, as for draws */
   etna_set_state(stream, VIVS_PS_NEWRANGE_LOW, 0);
   etna_set_state(stream, VIVS_PS_HALTI5_RANGE_HIGH, num_inst);
   etna_set_state_reloc(stream, VIVS_PS_INST_ADDR, &(struct etna_reloc) {
      .bo = v->bo,
      .flags = ETNA_RELOC_READ,
   });
   etna_set_state(stream, VIVS_SH_CONFIG, VIVS_SH_CONFIG_RTNE_ROUNDING);
   etna_set_state(stream, VIVS_SH_ICACHE_CONTROL, VIVS_SH_ICACHE_CONTROL_ENABLE);
   etna_set_state(stream, VIVS_PS_ICACHE_COUNT, num_inst - 1);
   etna_set_state(stream, VIVS_VS_ICACHE_INVALIDATE, 0x1f);
   etna_set_state(stream, VIVS_PS_INPUT_COUNT,
                  VIVS_PS_INPUT_COUNT_COUNT(input_count) |
                  VIVS_PS_INPUT_COUNT_UNK8(31));
   etna_set_state(stream, VIVS_PS_TEMP_REGISTER_CONTROL,
                  VIVS_PS_TEMP_REGISTER_CONTROL_NUM_TEMPS(num_temps));
   etna_set_state(stream, VIVS_PS_CONTROL, VIVS_PS_CONTROL_RT_COUNT(7) | 0x800);
   etna_set_state(stream, VIVS_VS_VERTEX_CACHE_CONFIG, 0);
   etna_set_state(stream, VIVS_PS_PSCS_THROTTLE, 0);
   etna_set_state(stream, VIVS_PA_ATTRIBUTE_ELEMENT_COUNT,
                  VIVS_PA_ATTRIBUTE_ELEMENT_COUNT_COUNT(1));
   etna_set_state(stream, VIVS_GL_VARYING_TOTAL_COMPONENTS, 0);
   etna_set_state(stream, VIVS_PS_CONTROL_EXT, 0);
   etna_set_state(stream, VIVS_VS_LOAD_BALANCING, 0x0f3f0000);
   etna_set_state(stream, VIVS_VS_OUTPUT_COUNT, 1);
   etna_set_state(stream, VIVS_PA_VS_OUTPUT_COUNT, 0);
   etna_set_state(stream, VIVS_GL_HALTI5_SH_SPECIALS,
                  VIVS_GL_HALTI5_SH_SPECIALS_UNK16(0x7f));
   etna_set_state(stream, VIVS_PS_ICACHE_PREFETCH, 0);
   etna_set_state(stream, VIVS_GL_SHADER_INDEX, 0);

   etna_set_state(stream, VIVS_CL_CONFIG,
                  VIVS_CL_CONFIG_DIMENSIONS(3) | VIVS_CL_CONFIG_VALUE_ORDER(3));
   etna_set_state(stream, VIVS_CL_UNK00924, ctx->compute_unk924);
   etna_set_state(stream, VIVS_CL_THREAD_ALLOCATION, thread_alloc);
   etna_set_state(stream, VIVS_CL_GLOBAL_SCALE_X, 1);
   etna_set_state(stream, VIVS_CL_GLOBAL_SCALE_Y, 1);
   etna_set_state(stream, VIVS_CL_GLOBAL_SCALE_Z, 1);

   /* the uniforms of the first workgroup of a part (etna_compute_parts()) */
   const uint32_t uniform_base = ctx->screen->specs.ps_uniforms_offset;

   for (uint32_t z0 = 0; z0 < grid[2]; z0 += part[2]) {
   for (uint32_t y0 = 0; y0 < grid[1]; y0 += part[1]) {
   for (uint32_t x0 = 0; x0 < grid[0]; x0 += part[0]) {
   const uint32_t first[3] = { x0, y0, z0 };

   /* the hardware offsets the global ID, the shader the workgroup ID */
   if (parts > 1) {
      for (unsigned i = 0; i < v->uniforms.count; i++) {
         if (v->uniforms.contents[i] == ETNA_UNIFORM_BASE_WORKGROUP_ID)
            etna_set_state(stream, uniform_base + i * 4, first[v->uniforms.data[i]]);
      }
   }
   etna_set_state(stream, VIVS_CL_GLOBAL_WORK_OFFSET_X, x0 * hw_block[0]);
   etna_set_state(stream, VIVS_CL_GLOBAL_WORK_OFFSET_Y, y0 * hw_block[1]);
   etna_set_state(stream, VIVS_CL_GLOBAL_WORK_OFFSET_Z, z0 * hw_block[2]);

   /* CL_WORKGROUP_COUNT_{X,Y,Z} and CL_WORKGROUP_SIZE_{X,Y,Z}, minus one */
   const uint32_t groups[6] = {
      MIN2(part[0], grid[0] - x0) - 1,
      MIN2(part[1], grid[1] - y0) - 1,
      MIN2(part[2], grid[2] - z0) - 1,
      (hw_block[0] - 1) & 0x3ff,
      (hw_block[1] - 1) & 0x3ff,
      (hw_block[2] - 1) & 0x3ff,
   };
   etna_set_state_multi(stream, VIVS_CL_WORKGROUP_COUNT_X, 6, groups);

   etna_set_state(stream, VIVS_CL_KICKER, 0xbadabeeb);

   /* Wait for the dispatch before the state of the next one: the FE->PE
    * stall alone can let the last invocations of a dispatch run on into the
    * next dispatch's uniforms (their stores land in the next buffer range).
    * A TS flush before it, as NXP's driver has between dispatches and the
    * kernel at the end of a job, makes the stall wait for them. */
   etna_set_state(stream, VIVS_GL_FLUSH_CACHE,
                  VIVS_GL_FLUSH_CACHE_TEXTURE | VIVS_GL_FLUSH_CACHE_SHADER_L1 |
                  VIVS_GL_FLUSH_CACHE_UNK10 | VIVS_GL_FLUSH_CACHE_UNK11);
   if (!VIV_FEATURE(ctx->screen, ETNA_FEATURE_BLT_ENGINE))
      etna_set_state(stream, VIVS_TS_FLUSH_CACHE, VIVS_TS_FLUSH_CACHE_FLUSH);
   etna_stall(stream, SYNC_RECIPIENT_FE, SYNC_RECIPIENT_PE);
   }
   }
   }

   /* the linear base is now the newest copy of written images */
   u_foreach_bit(i, images->enabled_mask) {
      if ((images->img[i].access & PIPE_IMAGE_ACCESS_WRITE) &&
          images->img[i].resource->target != PIPE_BUFFER)
         etna_resource_level_mark_changed(
            &etna_resource(images->img[i].resource)->levels[images->img[i].u.tex.level]);
   }

   /* The dispatch clobbered the 3D shader state: start over at the next draw. */
   ctx->needs_gpu_state_reset = true;
   ctx->dirty = ~0L;

   if (DBG_ENABLED(ETNA_DBG_FLUSH_ALL))
      pctx->flush(pctx, NULL, 0);

out:
   if (textures)
      etna_compute_swap_textures(ctx, &saved, false, shader);
}

/* OpenCL global buffers: each handle holds an offset into its buffer, the
 * kernel gets the GPU address (32-bit, address_bits = 32) */
static void
etna_set_global_binding(struct pipe_context *pctx, unsigned first, unsigned count,
                        struct pipe_resource **resources, uint32_t **handles)
{
   struct etna_context *ctx = etna_context(pctx);

   assert(first + count <= ARRAY_SIZE(ctx->global_bindings));

   for (unsigned i = 0; i < count; i++) {
      const unsigned n = first + i;

      if (resources && resources[i]) {
         pipe_resource_reference(&ctx->global_bindings[n], resources[i]);
         ctx->global_bindings_mask |= BITFIELD_BIT(n);

         if (handles && handles[i]) {
            uint32_t offset;
            memcpy(&offset, handles[i], sizeof(offset));
            offset += etna_bo_gpu_va(etna_buffer_resource(resources[i])->bo);
            memcpy(handles[i], &offset, sizeof(offset));
         }
      } else {
         pipe_resource_reference(&ctx->global_bindings[n], NULL);
         ctx->global_bindings_mask &= ~BITFIELD_BIT(n);
      }
   }
}

void
etna_compute_init(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);

   if (!pctx->screen->caps.compute)
      return;

   ctx->compute_unk924 = debug_get_num_option("ETNA_CL_UNK924", 0x00100000);
   ctx->compute_thread_alloc = debug_get_num_option("ETNA_CL_THREAD_ALLOC", 0);
   ctx->compute_min_temps = debug_get_num_option("ETNA_CL_MIN_TEMPS", 1);

   pctx->create_compute_state = etna_create_compute_state;
   pctx->bind_compute_state = etna_bind_compute_state;
   pctx->delete_compute_state = etna_delete_compute_state;
   pctx->get_compute_state_info = etna_get_compute_state_info;
   pctx->launch_grid = etna_launch_grid;
   pctx->set_shader_buffers = etna_set_shader_buffers;
   pctx->set_shader_images = etna_set_shader_images;
   pctx->set_global_binding = etna_set_global_binding;
   pctx->memory_barrier = etna_memory_barrier;
}

void
etna_compute_fini(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);

   if (ctx->compute_shared_bo)
      etna_bo_del(ctx->compute_shared_bo);
   if (ctx->compute_scratch_bo)
      etna_bo_del(ctx->compute_scratch_bo);
   if (ctx->compute_private_bo)
      etna_bo_del(ctx->compute_private_bo);

   for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++)
      pipe_sampler_view_reference(&ctx->cs_sampler_view[i], NULL);

   if (ctx->cs_fetch_sampler)
      pctx->delete_sampler_state(pctx, ctx->cs_fetch_sampler);
   for (unsigned i = 0; i < ARRAY_SIZE(ctx->cs_sampler_copy); i++) {
      if (ctx->cs_sampler_copy[i])
         pctx->delete_sampler_state(pctx, ctx->cs_sampler_copy[i]);
   }

   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
      for (unsigned i = 0; i < ETNA_MAX_SHADER_BUFFERS; i++)
         pipe_resource_reference(&ctx->shader_buffers[s].sb[i].buffer, NULL);
      for (unsigned i = 0; i < ETNA_MAX_SHADER_IMAGES; i++)
         util_copy_image_view(&ctx->shader_images[s].img[i], NULL);
   }
}
