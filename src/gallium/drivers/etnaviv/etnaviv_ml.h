/*
 * Copyright (c) 2023-2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef H_ETNA_ML
#define H_ETNA_ML

#include <stdlib.h>
#include "pipe/p_state.h"
#include "util/u_dynarray.h"
#include "util/macros.h"
#include "etnaviv_context.h"

#define MAX_CONFIG_BOS 4

struct etna_ml_device {
   struct pipe_ml_device base;
};

static inline struct etna_ml_device *
etna_ml_device(struct pipe_ml_device *dev)
{
   return (struct etna_ml_device *)dev;
}

enum etna_job_type {
    ETNA_JOB_TYPE_NN,
    ETNA_JOB_TYPE_TP,
    ETNA_JOB_TYPE_CONCAT, /* Fake operation, won't execute on HW. Hack will go away after the move to NIR. */
    ETNA_JOB_TYPE_SPLIT, /* Fake operation, won't execute on HW. Hack will go away after the move to NIR. */
    ETNA_JOB_TYPE_BYPASS, /* Fake operation, won't execute on HW. Hack will go away after the move to NIR. */
};

enum etna_ml_tp_type {
   ETNA_ML_TP_TRANSPOSE,
   ETNA_ML_TP_DETRANSPOSE,
   ETNA_ML_TP_RESHUFFLE,
   ETNA_ML_TP_PAD,
   ETNA_ML_TP_RELU,
   ETNA_ML_TP_ABSOLUTE,
   ETNA_ML_TP_LOGISTIC,
   ETNA_ML_TP_MAXPOOL,
};

enum etna_ml_tensor_layout {
   ETNA_ML_LAYOUT_ANY = 0,
   ETNA_ML_LAYOUT_NHWC,
   ETNA_ML_LAYOUT_NCHW,
};

struct etna_ml_tensor {
   struct pipe_resource *resource;
   unsigned offset;
   unsigned size;
   enum etna_ml_tensor_layout exp_layout; /* expected */
   enum etna_ml_tensor_layout act_layout; /* actual */
   bool in_sram;          /* local experiment: lives in VIP SRAM at sram_offset */
   unsigned sram_offset;
};

/* Local experiment: a graph input that the CPU writes transposed (NHWC -> planar) into out_idx */
struct etna_cpu_transpose {
   unsigned in_idx, out_idx, h, w, c;
   unsigned stride, out_w, out_h, pad_x, pad_y; /* stride 2: also destride like the TP reshuffle */
   uint8_t zp;
};

/* Local experiment ETNA_CPU_DETRANSPOSE: a graph output that the NPU leaves planar, written back in
 * NHWC order (out[p * c + ch] = in[ch * pixels + p]) when it is copied out */
struct etna_cpu_detranspose {
   unsigned idx, pixels, c;
};

/* Local experiment: getenv once, for the per-inference paths */
#define ETNA_ML_ENV(name) ({ static int env_ = -1; if (env_ < 0) env_ = getenv(name) != NULL; env_; })

struct etna_ml_subgraph {
   struct pipe_ml_subgraph base;
   struct etna_screen *screen;

   struct util_dynarray operations;

   /* Indexed by tensor index */
   struct util_dynarray tensors; /* Contains struct etna_ml_tensor */

   struct etna_bo *sram_bo;  /* local experiment: reserves the NPU VA range remapped to VIP SRAM */
   uint32_t sram_window;
   bool listed_once; /* local experiment ETNA_ML_LIST_ONCE */
   struct util_dynarray cpu_transposes; /* local experiment: struct etna_cpu_transpose */
   struct util_dynarray cpu_detransposes; /* local experiment: struct etna_cpu_detranspose */
   uint32_t *cmd_cache;       /* local experiment ETNA_ML_CMD_CACHE: recorded job words */
   unsigned cmd_cache_words;
   struct util_dynarray cmd_bos; /* BOs a replayed submit still lists (CPU-visible, waited on) */
   unsigned prefix_iter; /* local experiment ETNA_PREFIX */
   int prefix_limit;
};

struct etna_vip_instruction {
   enum etna_job_type type;
   enum etna_ml_tp_type tp_type;

   struct etna_bo *configs[MAX_CONFIG_BOS];
   struct etna_bo *coefficients;
   struct etna_bo *pwl_lut;
   struct pipe_resource *input;
   unsigned input_offset;
   struct pipe_resource *output;
   unsigned output_offset;
   bool input_external, output_external; /* local experiment: graph input/output, the CPU touches it */

   struct etna_bo *kernel;
};

#define MAX_TENSORS 10
struct etna_operation {
   struct list_head link;

   enum etna_job_type type;
   enum etna_ml_tp_type tp_type;

   bool addition;
   bool depthwise;
   bool pointwise;
   bool fully_connected;
   bool pooling_first_pixel;
   bool pooling_max; /* local experiment: 2x2 max instead of first pixel */
   bool padding_same;
   bool relu;

   unsigned stride;
   unsigned sram_cache_base; /* local experiment: first SRAM byte the caches may use, 0 = default */
   bool cpu_transpose;       /* local experiment: done by the CPU at input time, no job */

   unsigned input_tensors[MAX_TENSORS];
   unsigned input_count;
   unsigned input_tensor_sizes[MAX_TENSORS];

   /* The following apply to the first input tensor only */
   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   uint8_t input_zero_point;
   float input_scale;

   unsigned output_tensors[MAX_TENSORS];
   unsigned output_count;
   unsigned output_tensor_sizes[MAX_TENSORS];

   /* The following apply to the first output tensor only */
   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   uint8_t output_zero_point;
   float output_scale;

   uint8_t *weight_tensor;
   unsigned weight_tensor_size;
   unsigned weight_width;
   unsigned weight_height;
   uint8_t weight_zero_point;
   float weight_scale;
   bool weight_signed;

   uint8_t addition_offset;

   uint8_t *bias_tensor;
   unsigned bias_tensor_size;

   unsigned pad_before_x;
   unsigned pad_after_x;
   unsigned pad_before_y;
   unsigned pad_after_y;
   unsigned pad_before_z;
   unsigned pad_after_z;
};

struct etna_idx_replacement {
   struct list_head link;
   unsigned idx_old;
   unsigned idx_new;
};

#define ML_DBG(fmt, ...)                                  \
   do {                                                   \
      if (DBG_ENABLED(ETNA_DBG_ML_MSGS))                  \
         _debug_printf(fmt, ##__VA_ARGS__);             \
   } while (0)

unsigned etna_ml_allocate_tensor(struct etna_ml_subgraph *subgraph);
void etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size);
struct etna_ml_tensor *etna_ml_get_tensor(struct etna_ml_subgraph *subgraph, unsigned idx);
struct pipe_resource *etna_ml_get_resource(struct etna_ml_subgraph *subgraph, unsigned idx);
unsigned etna_ml_get_offset(struct etna_ml_subgraph *subgraph, unsigned idx);
unsigned etna_ml_get_size(struct etna_ml_subgraph *subgraph, unsigned idx);

struct etna_bo *etna_ml_create_bo(struct etna_screen *screen, size_t size);

struct pipe_resource *etna_ml_create_resource(struct pipe_screen *pscreen, size_t size);

struct etna_core_npu_info *etna_ml_get_core_info(struct etna_screen *screen);

bool
etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation);

struct pipe_ml_subgraph *
etna_ml_subgraph_create(struct pipe_ml_device *pdevice,
                        const struct pipe_ml_operation *operations,
                        unsigned count);

void
etna_ml_subgraph_invoke(struct pipe_context *pctx, struct pipe_ml_subgraph *psubgraph,
                        unsigned inputs_count, unsigned input_idxs[], void *inputs[], bool is_signed[]);

void
etna_ml_subgraph_read_outputs(struct pipe_context *context, struct pipe_ml_subgraph *subgraph,
                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],
                              bool is_signed[]);

void
etna_ml_subgraph_destroy(struct pipe_ml_device *pdevice, struct pipe_ml_subgraph *subgraph);

#endif
