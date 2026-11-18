/*
 * Copyright (c) 2026 Etnaviv Project
 * SPDX-License-Identifier: MIT
 */

/*
 * Timestamp queries (OpenCL profiling; GL emulates time elapsed with two).
 *
 * Vivante GPUs have no timestamp counter that the command stream can write.
 * Instead a query point flushes the context with an out fence: the GPU has
 * passed the point when the fence of that submit signals, and the sync file
 * says when that was, in the kernel's CLOCK_MONOTONIC, the clock of
 * os_time_get_nano() and of the default pipe_screen::get_timestamp(). The
 * point also can't be earlier than the CPU time at which it was recorded.
 *
 * A flush per query point costs a submit, so the queries are only offered
 * with the experimental compute support (etna_init_screen_caps()).
 */

#include "util/libsync.h"
#include "util/os_time.h"
#include "util/u_memory.h"

#include "etnaviv_context.h"
#include "etnaviv_fence.h"
#include "etnaviv_query.h"

struct etna_time_query {
   struct etna_query base;
   uint64_t cpu;                     /* CPU time when the point was recorded */
   uint64_t time;                    /* when the GPU passed it, once known */
   struct pipe_fence_handle *fence;  /* the flush at the point, until then */
};

static inline struct etna_time_query *
etna_time_query(struct etna_query *q)
{
   return (struct etna_time_query *)q;
}

static void
etna_time_destroy_query(struct etna_context *ctx, struct etna_query *q)
{
   struct etna_time_query *tq = etna_time_query(q);
   struct pipe_screen *pscreen = ctx->base.screen;

   pscreen->fence_reference(pscreen, &tq->fence, NULL);
   FREE(tq);
}

static void
etna_time_begin_query(struct etna_context *ctx, struct etna_query *q)
{
}

static void
etna_time_end_query(struct etna_context *ctx, struct etna_query *q)
{
   struct etna_time_query *tq = etna_time_query(q);
   struct pipe_context *pctx = &ctx->base;

   /* etna_flush() sets the fence without releasing the old one */
   pctx->screen->fence_reference(pctx->screen, &tq->fence, NULL);
   tq->time = 0;
   tq->cpu = os_time_get_nano();
   pctx->flush(pctx, &tq->fence, PIPE_FLUSH_FENCE_FD);
}

/* when the fence signaled, from its sync file */
static uint64_t
fence_signal_time(struct pipe_screen *pscreen, struct pipe_fence_handle *fence)
{
   uint64_t t = 0;
   int fd = pscreen->fence_get_fd(pscreen, fence);

   if (fd >= 0) {
      struct sync_file_info *info = sync_file_info(fd);

      if (info) {
         const struct sync_fence_info *fi =
            (const struct sync_fence_info *)(uintptr_t)info->sync_fence_info;

         for (unsigned i = 0; i < info->num_fences; i++)
            t = MAX2(t, fi[i].timestamp_ns);
         free(info);
      }
      close(fd);
   }

   /* no sync file (the submit failed): now, after the wait */
   return t ? t : os_time_get_nano();
}

static bool
etna_time_get_query_result(struct etna_context *ctx, struct etna_query *q,
                           bool wait, union pipe_query_result *result)
{
   struct etna_time_query *tq = etna_time_query(q);
   struct pipe_screen *pscreen = ctx->base.screen;

   if (tq->fence) {
      if (pscreen->fence_finish(pscreen, &ctx->base, tq->fence,
                                wait ? OS_TIMEOUT_INFINITE : 0)) {
         tq->time = MAX2(tq->cpu, fence_signal_time(pscreen, tq->fence));
      } else if (etna_fence_failed(tq->fence)) {
         /* A job the kernel killed (hang check) is over too: rusticl fails
          * its event, and still reads the profiling times, with
          * read_blocked(), which panics without one. Give the time on the
          * CPU when the query ended. */
         tq->time = tq->cpu;
      } else {
         return false;
      }
      pscreen->fence_reference(pscreen, &tq->fence, NULL);
   }

   result->u64 = tq->time;

   return true;
}

static const struct etna_query_funcs time_query_funcs = {
   .destroy_query = etna_time_destroy_query,
   .begin_query = etna_time_begin_query,
   .end_query = etna_time_end_query,
   .get_query_result = etna_time_get_query_result,
};

struct etna_query *
etna_time_create_query(struct etna_context *ctx, unsigned query_type)
{
   if (query_type != PIPE_QUERY_TIMESTAMP || !ctx->base.screen->caps.query_timestamp)
      return NULL;

   struct etna_time_query *tq = CALLOC_STRUCT(etna_time_query);
   if (!tq)
      return NULL;

   tq->base.funcs = &time_query_funcs;
   tq->base.type = query_type;

   return &tq->base;
}
