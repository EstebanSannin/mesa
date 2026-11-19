/*
 * Copyright (c) 2023-2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include "pipe/p_state.h"
#include <sys/time.h>

#include "util/u_inlines.h"
#include "util/u_queue.h"

#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_emit.h"
#include "etnaviv_ml_nn.h"
#include "etnaviv_ml_tp.h"
#include "etnaviv_ml.h"
#include "etnaviv_resource.h"
#include "etnaviv_screen.h"

struct etna_ml_tensor *
etna_ml_get_tensor(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   struct etna_ml_tensor **tensor = util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor*, idx);
   if (*tensor == NULL)
      *tensor = calloc(1, sizeof(**tensor));
   return *tensor;
}

struct pipe_resource *
etna_ml_get_resource(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->resource;
}

unsigned
etna_ml_get_offset(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->offset;
}

unsigned
etna_ml_get_size(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   return etna_ml_get_tensor(subgraph, idx)->size;
}

static void
etna_ml_copy_layout(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned src_idx)
{
   struct etna_ml_tensor *dst = etna_ml_get_tensor(subgraph, idx);
   struct etna_ml_tensor *src = etna_ml_get_tensor(subgraph, src_idx);
   dst->exp_layout = src->exp_layout;
   dst->act_layout = src->act_layout;
}

unsigned
etna_ml_allocate_tensor(struct etna_ml_subgraph *subgraph)
{
   struct etna_ml_tensor **tensors = util_dynarray_grow(&subgraph->tensors, struct etna_ml_tensor *, 1);
   tensors[0] = calloc(1, sizeof(*tensors[0]));
   tensors[0]->resource = NULL;
   tensors[0]->offset = 0;
   tensors[0]->size = 0;

   return util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *) - 1;
}

void
etna_ml_create_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);

   assert(idx < util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *));
   assert(size > 0);

   struct pipe_resource *res = tensor->resource;

   if (res != NULL) {
      assert(size == tensor->size);
      return;
   }

   res = etna_ml_create_resource(&subgraph->screen->base, size);
   tensor->resource = res;
   tensor->size = size;

   ML_DBG("created resource %p for tensor %d with size %d\n", res, idx, size);
}

static void
etna_ml_destroy_tensor(struct etna_ml_subgraph *subgraph, unsigned idx)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);

   pipe_resource_reference(&tensor->resource, NULL);
   tensor->offset = 0;
   tensor->size = 0;
}

struct etna_bo *
etna_ml_create_bo(struct etna_screen *screen, size_t size)
{
   struct etna_bo *bo = etna_bo_new(screen->dev,
                                    size,
                                    DRM_ETNA_GEM_CACHE_WC);

   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);
   struct etna_nn_params *map = etna_bo_map(bo);
   memset(map, 0, size);
   etna_bo_cpu_fini(bo);

   return bo;
}

struct pipe_resource *
etna_ml_create_resource(struct pipe_screen *pscreen, size_t size)
{
   struct pipe_resource *res;

   /* Local experiment ETNA_ML_CACHED=1: cached tensors, so the CPU reads inputs and outputs at cache
    * speed instead of from write-combined memory (cache maintenance is cheap on arm64). */
   if (getenv("ETNA_ML_CACHED")) {
      struct pipe_resource templ = {
         .target = PIPE_BUFFER,
         .format = PIPE_FORMAT_R8_UNORM,
         .width0 = size,
         .height0 = 1,
         .depth0 = 1,
         .array_size = 1,
         .usage = PIPE_USAGE_DEFAULT,
         .flags = ETNA_RESOURCE_FLAG_CPU_READ,
      };
      res = pscreen->resource_create(pscreen, &templ);
   } else
      res = pipe_buffer_create(pscreen, 0, PIPE_USAGE_DEFAULT, size);

   struct etna_bo *bo = etna_buffer_resource(res)->bo;
   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);
   void *ptr = etna_bo_map(bo);
   memset(ptr, 0, pipe_buffer_size(res));
   etna_bo_cpu_fini(bo);

   return res;
}

struct etna_core_npu_info *
etna_ml_get_core_info(struct etna_screen *screen)
{
   struct etna_core_info *info = etna_gpu_get_core_info(screen->npu);
   return &info->npu;
}

static bool
needs_reshuffle(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation)
{
   unsigned nn_core_version = subgraph->screen->specs.nn_core_version;
   bool has_stride = poperation->conv.stride_x > 1 || poperation->conv.stride_y > 1;
   bool pointwise = poperation->conv.pointwise;
   bool has_padding = poperation->conv.padding_top > 0 ||
                      poperation->conv.padding_bottom > 0 ||
                      poperation->conv.padding_left > 0 ||
                      poperation->conv.padding_right > 0;
   unsigned input_width = poperation->input_tensors[0]->dims[1];

   if (!has_stride)
      return false;

   if (nn_core_version < 8)
      return !(poperation->conv.depthwise && (input_width > 5 || input_width < 3)) && !pointwise;
   else {
      unsigned input_channels = poperation->input_tensors[0]->dims[3];

      if (poperation->conv.depthwise)
         return false;

      if (poperation->conv.pointwise && input_width >= 3 && input_channels > 1)
         return false;

      if (poperation->conv.pointwise && has_padding)
         return false;

      return true;
   }
}

static const struct pipe_ml_operation *
etna_ml_find_producer(const struct pipe_ml_operation *poperations,
                      unsigned count,
                      unsigned tensor_idx)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->output_count; j++)
         if (poperation->output_tensors[j]->index == tensor_idx)
            return poperation;
   }

   return NULL;
}

static void
reference_tensor_with_offset(struct etna_ml_subgraph *subgraph,
                             unsigned src_tensor,
                             unsigned dst_tensor,
                             unsigned offset,
                             unsigned size)
{
   struct etna_ml_tensor *src = etna_ml_get_tensor(subgraph, src_tensor);
   struct etna_ml_tensor *dst = etna_ml_get_tensor(subgraph, dst_tensor);
   struct pipe_resource *old_res = dst->resource;
   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);
   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   ML_DBG("src_tensor %d (%x) dst_tensor %d offset %d size %d\n", src_tensor, etna_bo_gpu_va(etna_buffer_resource(src->resource)->bo), dst_tensor, offset, size);
   pipe_resource_reference(&dst->resource, src->resource);
   dst->offset = offset;
   dst->size = size;

   if (old_res) {
      for (int i = 0; i < num_tensors; i++) {
         if (etna_ml_get_resource(subgraph, i) == old_res) {
            pipe_resource_reference(&tensors[i]->resource, src->resource);
            tensors[i]->size = size;
            tensors[i]->offset = offset;
         }
      }
   }
}

static void
recreate_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct pipe_resource *old_res = etna_ml_get_resource(subgraph, idx);
   struct etna_ml_tensor **tensors = util_dynarray_begin(&subgraph->tensors);
   unsigned num_tensors = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   struct pipe_resource *new_res;

   etna_ml_destroy_tensor(subgraph, idx);
   etna_ml_create_tensor(subgraph, idx, size);
   new_res = etna_ml_get_resource(subgraph, idx);

   if (old_res) {
      for (int i = 0; i < num_tensors; i++) {
         if (etna_ml_get_resource(subgraph, i) == old_res) {
            pipe_resource_reference(&tensors[i]->resource, new_res);
            tensors[i]->size = size;
         }
      }
   }
}

static void
resize_tensor(struct etna_ml_subgraph *subgraph, unsigned idx, unsigned size)
{
   struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, idx);
   tensor->size = size;
}

static void
dump_graph(struct list_head *etna_operations)
{
   ML_DBG("\n");
   ML_DBG("dumping intermediate graph: %d operations\n", list_length(etna_operations));

   ML_DBG("\n");
   ML_DBG("%3s %-4s %3s %3s  %s\n", "idx", "type", "in", "out", "operation type-specific");
   ML_DBG("================================================================================================\n");
   unsigned i = 0;
   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      switch(operation->type) {
      case ETNA_JOB_TYPE_TP:
         ML_DBG("%3d %-4s %3d %3d",
                i, "TP", operation->input_tensors[0], operation->output_tensors[0]);
         break;
      case ETNA_JOB_TYPE_NN:
         ML_DBG("%3d %-4s %3d %3d in2: %3d",
                i, "NN", operation->input_tensors[0], operation->output_tensors[0], operation->input_tensors[1]);
         break;
      case ETNA_JOB_TYPE_CONCAT:
         ML_DBG("%3d %-4s %3d %3d in2: %3d",
                i, "CONC", operation->input_tensors[0], operation->output_tensors[0], operation->input_tensors[1]);
         break;
      case ETNA_JOB_TYPE_SPLIT:
         ML_DBG("%3d %-4s %3d %3d out2: %3d",
                i, "SPLIT", operation->input_tensors[0], operation->output_tensors[0], operation->output_tensors[1]);
         break;
      case ETNA_JOB_TYPE_BYPASS:
         ML_DBG("%3d %-4s %3d %3d",
                i, "BYPASS", operation->input_tensors[0], operation->output_tensors[0]);
         break;
      }
      ML_DBG("\n");
      i++;
   }
   ML_DBG("\n");
}

static void
add_bypass(const struct pipe_ml_operation *poperation, unsigned input_tensor,
           struct etna_operation *operation, struct list_head *etna_operations)
{
   operation->type = ETNA_JOB_TYPE_BYPASS;
   operation->input_count = 1;
   operation->input_tensors[0] = input_tensor;
   operation->input_tensor_sizes[0] = poperation->input_tensors[0]->dims[1] *
                                      poperation->input_tensors[0]->dims[2] *
                                      poperation->input_tensors[0]->dims[3];
   operation->output_count = 1;
   operation->output_tensors[0] = poperation->output_tensors[0]->index;
   operation->output_tensor_sizes[0] = poperation->output_tensors[0]->dims[1] *
                                       poperation->output_tensors[0]->dims[2] *
                                       poperation->output_tensors[0]->dims[3];

   list_addtail(&operation->link, etna_operations);
}

static bool
is_3d(struct pipe_tensor *tensor)
{
   return tensor->dims[1] > 1 &&
          tensor->dims[2] > 1 &&
          tensor->dims[3] > 1;
}

/** Tensor layout inference:
  * - Graph inputs are in NHWC order.
  * - Graph outputs are expected in NHWC order.
  * - Element-wise operations don't care about the layout.
  * - Other operations expect the tensors in NCHW order (if input_channels > 1) and their outputs are in NCHW order.
  * - Implicit transposes and detransposes are the only operations that change channel order.
  * - Explicit transposes and detransposes are ignored.
  */

/* A global average pooling (VALID padding, filter as large as the input, 1x1 output) is a depthwise
 * convolution with all weights 1/N, which is how the blob runs it. Pooling with padding can't be
 * expressed that way, as TFLite divides by the number of valid pixels at the borders. */
static bool
is_global_avg_pool(const struct pipe_ml_operation *poperation)
{
   const struct pipe_tensor *in = poperation->input_tensors[0];
   const struct pipe_tensor *out = poperation->output_tensors[0];

   return poperation->type == PIPE_ML_OPERATION_TYPE_POOLING &&
          poperation->pooling.type == PIPE_ML_POOLING_TYPE_AVG &&
          poperation->pooling.padding_top == 0 && poperation->pooling.padding_bottom == 0 &&
          poperation->pooling.padding_left == 0 && poperation->pooling.padding_right == 0 &&
          poperation->pooling.filter_height == in->dims[1] &&
          poperation->pooling.filter_width == in->dims[2] &&
          out->dims[1] == 1 && out->dims[2] == 1 && in->dims[3] == out->dims[3];
}

/* 2x2 stride-2 max pooling without padding: the NN computes a depthwise 1x1 identity convolution and pools its
 * output 2x2 with MAX, the way stride-2 depthwise convolutions use first-pixel pooling (the blob does the same). */
static bool
is_maxpool_2x2(const struct pipe_ml_operation *poperation)
{
   return getenv("ETNA_MP2_NN") && poperation->type == PIPE_ML_OPERATION_TYPE_POOLING &&
          poperation->pooling.type == PIPE_ML_POOLING_TYPE_MAX &&
          poperation->pooling.filter_width == 2 && poperation->pooling.filter_height == 2 &&
          poperation->pooling.stride_x == 2 && poperation->pooling.stride_y == 2 &&
          poperation->pooling.padding_top == 0 && poperation->pooling.padding_bottom == 0 &&
          poperation->pooling.padding_left == 0 && poperation->pooling.padding_right == 0;
}

static void
avg_pool_as_conv(const struct pipe_ml_operation *poperation, struct pipe_ml_operation *conv,
                 struct pipe_tensor *weight, struct pipe_tensor *bias)
{
   const struct pipe_tensor *in = poperation->input_tensors[0];
   const struct pipe_tensor *out = poperation->output_tensors[0];
   bool max2 = is_maxpool_2x2(poperation);
   unsigned kh = max2 ? 1 : in->dims[1], kw = max2 ? 1 : in->dims[2], channels = in->dims[3];

   memset(weight, 0, sizeof(*weight));
   weight->dims[0] = 1;
   weight->dims[1] = kh;
   weight->dims[2] = kw;
   weight->dims[3] = channels;
   weight->rank = 4;
   weight->scale = max2 ? out->scale / in->scale : 1.0f / (kh * kw);
   weight->is_constant = true;
   weight->data = malloc(kh * kw * channels);
   memset(weight->data, 1, kh * kw * channels);

   memset(bias, 0, sizeof(*bias));
   bias->dims[0] = 1;
   bias->dims[1] = 1;
   bias->dims[2] = 1;
   bias->dims[3] = channels;
   bias->rank = 4;
   bias->scale = in->scale * weight->scale;
   bias->is_constant = true;
   bias->data = calloc(channels, sizeof(int32_t));

   *conv = *poperation;
   memset(&conv->conv, 0, sizeof(conv->conv));
   conv->type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
   conv->conv.weight_tensor = weight;
   conv->conv.bias_tensor = bias;
   conv->conv.stride_x = max2 ? 2 : 1;
   conv->conv.stride_y = max2 ? 2 : 1;
   conv->conv.depthwise = true;
   conv->conv.activation_min = out->is_signed ? INT8_MIN : 0;
   conv->conv.activation_max = out->is_signed ? INT8_MAX : UINT8_MAX;
   conv->conv.dilation_width_factor = 1;
   conv->conv.dilation_height_factor = 1;
}

static void
lower_operations(struct etna_ml_subgraph *subgraph,
                 const struct pipe_ml_operation *poperations,
                 unsigned count,
                 struct list_head *etna_operations)
{
   struct list_head input_replacements;

   list_inithead(&input_replacements);

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      struct etna_operation *operation = calloc(1, sizeof(*operation));
      struct pipe_ml_operation pool_conv;
      struct pipe_tensor pool_weight, pool_bias;

      bool pool_max2 = is_maxpool_2x2(poperation);
      if (is_global_avg_pool(poperation) || pool_max2) {
         ML_DBG("Lowering pooling to a depthwise convolution.\n");
         avg_pool_as_conv(poperation, &pool_conv, &pool_weight, &pool_bias);
         poperation = &pool_conv;
      }
      assert(poperation->input_count <= MAX_TENSORS);
      unsigned input_tensors[MAX_TENSORS] = {};

      for (int i = 0; i < poperation->input_count; i++) {
         struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, poperation->input_tensors[i]->index);
         enum etna_ml_tensor_layout operation_layout = ETNA_ML_LAYOUT_ANY;

         if (poperation->type == PIPE_ML_OPERATION_TYPE_CONVOLUTION ||
             poperation->type == PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED ||
             poperation->type == PIPE_ML_OPERATION_TYPE_CONCATENATION ||
             poperation->type == PIPE_ML_OPERATION_TYPE_POOLING)
            operation_layout = ETNA_ML_LAYOUT_NCHW;

         if (!etna_ml_find_producer(poperations, count, poperation->input_tensors[i]->index)) {
            /* In TensorFlow Lite, graph inputs are in channel-last */
            tensor->act_layout = ETNA_ML_LAYOUT_NHWC;
            tensor->exp_layout = ETNA_ML_LAYOUT_NHWC;
         }

         input_tensors[i] = poperation->input_tensors[i]->index;

         if (poperation->input_tensors[i]->data != NULL)
            continue;

         if (operation_layout != ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout != operation_layout) {
            ML_DBG("Adding transpose.\n");
            struct etna_operation *transpose = calloc(1, sizeof(*transpose));
            etna_ml_lower_transpose(subgraph, poperation->input_tensors[i], transpose);
            transpose->input_tensors[0] = input_tensors[i];
            transpose->output_tensors[0] = etna_ml_allocate_tensor(subgraph);
            input_tensors[i] = transpose->output_tensors[0];
            list_addtail(&transpose->link, etna_operations);

            if (getenv("ETNA_CPU_TRANSPOSE") &&
                !etna_ml_find_producer(poperations, count, poperation->input_tensors[i]->index)) {
               struct etna_cpu_transpose ct = {
                  .in_idx = transpose->input_tensors[0],
                  .out_idx = transpose->output_tensors[0],
                  .h = transpose->input_width,
                  .w = transpose->input_height,
                  .c = transpose->input_channels,
               };
               transpose->cpu_transpose = true;
               util_dynarray_append(&subgraph->cpu_transposes, ct);
               ML_DBG("Graph input %u transposed on the CPU into tensor %u.\n", ct.in_idx, ct.out_idx);
            }

            struct etna_ml_tensor *transposed_tensor = etna_ml_get_tensor(subgraph, input_tensors[i]);
            transposed_tensor->exp_layout = tensor->exp_layout;
            transposed_tensor->act_layout = operation_layout;

            struct etna_idx_replacement *replacement = calloc(1, sizeof(*replacement));
            replacement->idx_old = transpose->input_tensors[0];
            replacement->idx_new = transpose->output_tensors[0];
            list_addtail(&replacement->link, &input_replacements);
         }

         struct etna_ml_tensor *tensor2 = etna_ml_get_tensor(subgraph, input_tensors[i]);
         ML_DBG("operation %d input tensor %d layouts %d %d.\n", poperation->type, input_tensors[i], tensor2->exp_layout, tensor2->act_layout);
      }

      switch(poperation->type) {
         case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
            if (needs_reshuffle(subgraph, poperation)) {
               ML_DBG("Adding reshuffle for convolution operation.\n");
               struct etna_operation *reshuffle = calloc(1, sizeof(*reshuffle));
               etna_ml_lower_reshuffle(subgraph, poperation, reshuffle);
               reshuffle->input_tensors[0] = input_tensors[0];
               reshuffle->output_tensors[0] = etna_ml_allocate_tensor(subgraph);
               input_tensors[0] = reshuffle->output_tensors[0];
               list_addtail(&reshuffle->link, etna_operations);

               util_dynarray_foreach(&subgraph->cpu_transposes, struct etna_cpu_transpose, ct) {
                  if (getenv("ETNA_CPU_RESHUFFLE") && ct->out_idx == reshuffle->input_tensors[0] &&
                      !ct->stride && reshuffle->stride == 2) {
                     /* Same layout as create_reshuffle_config(): x along W, planes c * 4 + py * 2 + px */
                     ct->stride = 2;
                     ct->out_idx = reshuffle->output_tensors[0];
                     ct->out_w = reshuffle->output_height;
                     ct->out_h = reshuffle->output_width;
                     ct->pad_x = reshuffle->padding_same ? (reshuffle->weight_width - 2 + reshuffle->input_width % 2) / 2 : 0;
                     ct->pad_y = reshuffle->padding_same ? (reshuffle->weight_height - 2 + reshuffle->input_height % 2) / 2 : 0;
                     ct->zp = reshuffle->input_zero_point;
                     reshuffle->cpu_transpose = true;
                     ML_DBG("Reshuffle of graph input %u done on the CPU: %ux%u -> %ux%ux%u pad %u,%u\n", ct->in_idx,
                            ct->w, ct->h, ct->out_w, ct->out_h, ct->c * 4, ct->pad_x, ct->pad_y);
                  }
               }
               etna_ml_copy_layout(subgraph, reshuffle->output_tensors[0], reshuffle->input_tensors[0]);
            }

            ML_DBG("Adding convolution.\n");
            etna_ml_lower_convolution(subgraph, poperation, operation);
            if (poperation == &pool_conv) {
               free(pool_weight.data);
               free(pool_bias.data);
               if (pool_max2) {
                  assert(operation->pooling_first_pixel);
                  operation->pooling_max = true;
               }
            }
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            input_tensors[0] = operation->output_tensors[0];
            list_addtail(&operation->link, etna_operations);
            etna_ml_copy_layout(subgraph, operation->output_tensors[0], operation->input_tensors[0]);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_ADD: {
            etna_ml_lower_add(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->input_tensors[1] = input_tensors[1];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
            operation->type = ETNA_JOB_TYPE_CONCAT;
            assert(poperation->input_count <= MAX_TENSORS);
            for (int i = 0; i < poperation->input_count; i++) {
               operation->input_tensors[i] = input_tensors[i];
               operation->input_tensor_sizes[i] = poperation->input_tensors[i]->dims[1] *
                                                  poperation->input_tensors[i]->dims[2] *
                                                  poperation->input_tensors[i]->dims[3];
            }
            operation->input_count = poperation->input_count;

            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            operation->output_width = poperation->output_tensors[0]->dims[1];
            operation->output_height = poperation->output_tensors[0]->dims[2];
            operation->output_channels = poperation->output_tensors[0]->dims[3];
            operation->output_tensor_sizes[0] = operation->output_width *
                                                operation->output_height *
                                                operation->output_channels;

            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_SPLIT: {
            operation->type = ETNA_JOB_TYPE_SPLIT;

            operation->input_tensors[0] = poperation->input_tensors[1]->index;
            operation->input_tensor_sizes[0] = poperation->input_tensors[1]->dims[1] *
                                               poperation->input_tensors[1]->dims[2] *
                                               poperation->input_tensors[1]->dims[3];

            assert(poperation->output_count <= MAX_TENSORS);
            for (int i = 0; i < poperation->output_count; i++) {
               operation->output_tensors[i] = poperation->output_tensors[i]->index;
               operation->output_tensor_sizes[i] = poperation->output_tensors[i]->dims[1] *
                                                   poperation->output_tensors[i]->dims[2] *
                                                   poperation->output_tensors[i]->dims[3];
            }
            operation->output_count = poperation->output_count;

            list_addtail(&operation->link, etna_operations);

            break;
         }
         case PIPE_ML_OPERATION_TYPE_PAD: {
            ML_DBG("Adding pad operation.\n");
            etna_ml_lower_pad(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
            etna_ml_lower_fully_connected(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_RESHAPE: {
            /* A reshape is an alias of its input buffer. When its output is returned to TFLite, the data must be
             * in NHWC order already: the output shape (e.g. SSD box/class heads, [1, H*W*A, K]) is not 3D, so
             * the detranspose below for graph outputs would not catch it. NHWC [1, H, W, C] and the reshaped
             * tensor have the same byte order, so detranspose the 4D input and keep the alias. */
            struct etna_ml_tensor *reshape_in = etna_ml_get_tensor(subgraph, input_tensors[0]);
            if (poperation->output_tensors[0]->is_external_output &&
                reshape_in->act_layout == ETNA_ML_LAYOUT_NCHW &&
                is_3d(poperation->input_tensors[0]) &&
                getenv("ETNA_CPU_DETRANSPOSE")) {
               /* Local experiment: the output stays planar in DDR (the SRAM planner skips graph outputs),
                * read_outputs reorders it */
               struct etna_cpu_detranspose cd = {
                  .idx = poperation->output_tensors[0]->index,
                  .pixels = poperation->input_tensors[0]->dims[1] * poperation->input_tensors[0]->dims[2],
                  .c = poperation->input_tensors[0]->dims[3],
               };
               util_dynarray_append(&subgraph->cpu_detransposes, cd);
               ML_DBG("Graph output %u detransposed on the CPU (%u pixels, %u channels).\n", cd.idx, cd.pixels, cd.c);
            } else if (poperation->output_tensors[0]->is_external_output &&
                       reshape_in->act_layout == ETNA_ML_LAYOUT_NCHW &&
                       is_3d(poperation->input_tensors[0])) {
               ML_DBG("Adding detranspose before reshape of graph output %d.\n", poperation->output_tensors[0]->index);
               struct etna_operation *detranspose = calloc(1, sizeof(*detranspose));
               etna_ml_lower_detranspose(subgraph, poperation->input_tensors[0], detranspose);
               detranspose->input_tensors[0] = input_tensors[0];
               detranspose->output_tensors[0] = etna_ml_allocate_tensor(subgraph);
               list_addtail(&detranspose->link, etna_operations);

               struct etna_ml_tensor *nhwc = etna_ml_get_tensor(subgraph, detranspose->output_tensors[0]);
               nhwc->exp_layout = ETNA_ML_LAYOUT_NHWC;
               nhwc->act_layout = ETNA_ML_LAYOUT_NHWC;
               input_tensors[0] = detranspose->output_tensors[0];
            }
            add_bypass(poperation, input_tensors[0], operation, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_POOLING: {
            /* Global average pooling was turned into a convolution above, this is max pooling */
            ML_DBG("Adding max pooling.\n");
            etna_ml_lower_maxpool(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_RELU: {
            etna_ml_lower_relu(subgraph, poperation, poperation->input_tensors[0], operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_ABSOLUTE: {
            etna_ml_lower_absolute(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_LOGISTIC: {
            etna_ml_lower_logistic(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_SUBTRACT: {
            struct etna_operation *operation = calloc(1, sizeof(*operation));
            etna_ml_lower_add(subgraph, poperation, operation);
            operation->input_tensors[0] = input_tensors[0];
            operation->input_tensors[1] = input_tensors[1];
            operation->output_tensors[0] = poperation->output_tensors[0]->index;
            list_addtail(&operation->link, etna_operations);
            break;
         }
         case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
            add_bypass(poperation, input_tensors[0], operation, etna_operations);
            struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, operation->output_tensors[0]);
            tensor->act_layout = ETNA_ML_LAYOUT_NCHW;
            tensor->exp_layout = ETNA_ML_LAYOUT_NCHW;

            break;
         }
         default:
            UNREACHABLE("Unsupported ML operation type");
      }

      for (int i = 0; i < poperation->output_count; i++) {
         struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, poperation->output_tensors[i]->index);
         if (tensor->exp_layout == ETNA_ML_LAYOUT_ANY &&
             tensor->act_layout == ETNA_ML_LAYOUT_ANY) {
            ML_DBG("Copying layout to output tensor %d.\n", poperation->output_tensors[i]->index);
            etna_ml_copy_layout(subgraph, poperation->output_tensors[i]->index, operation->input_tensors[0]);
         }

         ML_DBG("type %d i %d tensor %d layout %d == %d\n", poperation->type, i, poperation->output_tensors[i]->index, tensor->exp_layout, tensor->act_layout);
         if (poperation->output_tensors[i]->is_external_output &&
             is_3d(poperation->output_tensors[i]) &&
             tensor->exp_layout != tensor->act_layout) {
            ML_DBG("Adding detranspose.\n");
            struct etna_operation *detranspose = calloc(1, sizeof(*detranspose));
            etna_ml_lower_detranspose(subgraph, poperation->output_tensors[i], detranspose);
            operation->output_tensors[i] = etna_ml_allocate_tensor(subgraph);
            detranspose->input_tensors[0] = operation->output_tensors[i];
            detranspose->output_tensors[0] = poperation->output_tensors[i]->index;
            list_addtail(&detranspose->link, etna_operations);

            struct etna_idx_replacement *replacement = calloc(1, sizeof(*replacement));
            replacement->idx_old = poperation->output_tensors[i]->index;
            replacement->idx_new = operation->output_tensors[i];
            list_addtail(&replacement->link, &input_replacements);
         }
      }
   }

   /*
    * This can not be folded into the above loop: if there is an Add operation
    * and then later a Conv2d operation that use the same input tensor, the
    * Add's input will have to be replaced as well when a Transpose was inserted
    * before the Conv2d.
    */
   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      /* The inserted transposes read the original tensor. When it has several consumers, each one gets its
       * own transpose, and must not be redirected to the output of another one. */
      if (operation->type == ETNA_JOB_TYPE_TP && operation->tp_type == ETNA_ML_TP_TRANSPOSE)
         continue;

      list_for_each_entry(struct etna_idx_replacement, replacement, &input_replacements, link) {
         if (operation->output_tensors[0] != replacement->idx_new) {
            for (int i = 0; i < operation->input_count; i++) {
               if (operation->input_tensors[i] == replacement->idx_old)
                  operation->input_tensors[i] = replacement->idx_new;
            }
         }
      }
   }

   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      if (operation->type == ETNA_JOB_TYPE_CONCAT) {
         etna_ml_create_tensor(subgraph, operation->output_tensors[0], operation->output_tensor_sizes[0]);

         unsigned offset = 0;
         for (int i = 0; i < operation->input_count; i++) {
            reference_tensor_with_offset(subgraph,
                                       operation->output_tensors[0],
                                       operation->input_tensors[i],
                                       offset,
                                       operation->input_tensor_sizes[i]);
            offset += operation->input_tensor_sizes[i];
         }
      } else if (operation->type == ETNA_JOB_TYPE_SPLIT) {
         etna_ml_create_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0]);

         unsigned offset = 0;
         for (int i = 0; i < operation->output_count; i++) {
            reference_tensor_with_offset(subgraph,
                                         operation->input_tensors[0],
                                         operation->output_tensors[i],
                                         offset,
                                         operation->output_tensor_sizes[i]);
            offset += operation->output_tensor_sizes[i];
         }
      } else if (operation->type == ETNA_JOB_TYPE_NN && operation->input_count > 1) { /* Add or Subtraction */
         recreate_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0] +
                                                                        operation->input_tensor_sizes[1]);
         resize_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0]);
         reference_tensor_with_offset(subgraph,
                                      operation->input_tensors[0],
                                      operation->input_tensors[1],
                                      operation->input_tensor_sizes[0],
                                      operation->input_tensor_sizes[1]);
      } else if (operation->type == ETNA_JOB_TYPE_BYPASS) {
         etna_ml_create_tensor(subgraph, operation->input_tensors[0], operation->input_tensor_sizes[0]);
         reference_tensor_with_offset(subgraph,
                                      operation->input_tensors[0],
                                      operation->output_tensors[0],
                                      0,
                                      operation->output_tensor_sizes[0]);
      } else {
         for (int i = 0; i < operation->input_count; i++)
            etna_ml_create_tensor(subgraph, operation->input_tensors[i], operation->input_tensor_sizes[i]);
      }
   }

   /* Create any output tensors that aren't inputs to other operations, these
    * are the outputs of the graph.
    */
   ML_DBG("Ensuring all output tensors have their memory backing.\n");
   list_for_each_entry(struct etna_operation, operation, etna_operations, link) {
      struct pipe_resource *res = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
      if (res != NULL)
         continue;

      etna_ml_create_tensor(subgraph, operation->output_tensors[0], operation->output_tensor_sizes[0]);
   }

   list_for_each_entry_safe(struct etna_idx_replacement, replacement, &input_replacements, link)
      free(replacement);

   if (DBG_ENABLED(ETNA_DBG_ML_MSGS))
      dump_graph(etna_operations);
}

static unsigned
count_tensors(const struct pipe_ml_operation *poperations,
              unsigned count)
{
   unsigned tensor_count = 0;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++)
         tensor_count = MAX2(tensor_count, poperation->input_tensors[j]->index);

      for (unsigned j = 0; j < poperation->output_count; j++)
         tensor_count = MAX2(tensor_count, poperation->output_tensors[j]->index);

      switch (poperation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         tensor_count = MAX2(tensor_count, poperation->conv.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->conv.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED:
         tensor_count = MAX2(tensor_count, poperation->fcon.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->fcon.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_PAD:
      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD:
      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
      case PIPE_ML_OPERATION_TYPE_SPLIT:
      case PIPE_ML_OPERATION_TYPE_RESHAPE:
      case PIPE_ML_OPERATION_TYPE_RELU:
      case PIPE_ML_OPERATION_TYPE_ABSOLUTE:
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
      case PIPE_ML_OPERATION_TYPE_TRANSPOSE:
      case PIPE_ML_OPERATION_TYPE_POOLING:
         break;
      default:
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   return tensor_count + 1;
}

static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}

static bool
convolution_activation_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *output = operation->output_tensors[0];
   int max = output->is_signed ? INT8_MAX : UINT8_MAX;

   return operation->conv.activation_max >= max;
}

bool
etna_ml_operation_supported(struct pipe_ml_device *pdevice,
                            const struct pipe_ml_operation *operation)
{
   bool supported = false;

   switch (operation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
         struct pipe_tensor *input_tensor = operation->input_tensors[0];
         struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
         struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
         struct pipe_tensor *output_tensor = operation->output_tensors[0];

         // Dilation and per-axis quantization not yet implemented
         /* Local experiment ETNA_PER_AXIS=1: per-channel weights are requantized to per-tensor at compile time
          * (etna_ml_lower_convolution), like the blob does for most layers. */
         bool per_axis_ok = getenv("ETNA_PER_AXIS") != NULL;
         if (tensor_quantization_supported(input_tensor) &&
             (per_axis_ok || tensor_quantization_supported(weight_tensor)) &&
             (per_axis_ok || tensor_quantization_supported(bias_tensor)) &&
             tensor_quantization_supported(output_tensor) &&
             operation->conv.dilation_width_factor == 1 &&
             operation->conv.dilation_height_factor == 1 &&
             convolution_activation_supported(operation)) {
            supported = true;
         }
         break;
      }
      case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      case PIPE_ML_OPERATION_TYPE_ADD: {
         supported = operation->input_tensors[0]->data == NULL &&
                     operation->input_tensors[1]->data == NULL;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
         supported = true;

         if (operation->conc.axis != 3 &&
             operation->conc.axis != -1)
            supported = false;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_SPLIT: {
         supported = true;

         if (operation->conc.axis != 3 &&
             operation->conc.axis != -1)
            supported = false;

         unsigned output_channels = operation->output_tensors[0]->dims[3];
         for (unsigned i = 1; i < operation->output_count; i++)
            if (output_channels != operation->output_tensors[i]->dims[3])
               supported = false;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_PAD: {
         supported = operation->pad.before_x <= 2 &&
                     operation->pad.after_x <= 2 &&
                     operation->pad.before_y <= 2 &&
                     operation->pad.after_y <= 2 &&
                     operation->pad.before_z <= 2 &&
                     operation->pad.after_z <= 2;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
         supported = operation->input_tensors[0]->dims[3] < 1280;
         break;
      }
      case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 3 &&
             operation->transpose.perm[2] == 1 &&
             operation->transpose.perm[3] == 2)
            supported = true;  /* Channels-last to channels-first */
         if (operation->transpose.perm[0] == 0 &&
             operation->transpose.perm[1] == 2 &&
             operation->transpose.perm[2] == 3 &&
             operation->transpose.perm[3] == 1)
            supported = true;  /* Channels-first to channels-last */
         break;
      }
      case PIPE_ML_OPERATION_TYPE_POOLING:
         if (operation->pooling.type == PIPE_ML_POOLING_TYPE_MAX) {
            supported = getenv("ETNA_MAXPOOL") &&
                        operation->pooling.filter_width == operation->pooling.filter_height &&
                        operation->pooling.stride_x == operation->pooling.stride_y &&
                        ((operation->pooling.filter_width == 3 &&
                          (operation->pooling.stride_x == 1 || operation->pooling.stride_x == 2)) ||
                         is_maxpool_2x2(operation) ||
                         (operation->pooling.filter_width == 2 && operation->pooling.stride_x == 2 &&
                          operation->pooling.padding_top == 0 && operation->pooling.padding_bottom == 0 &&
                          operation->pooling.padding_left == 0 && operation->pooling.padding_right == 0)) &&
                        tensor_quantization_supported(operation->input_tensors[0]) &&
                        tensor_quantization_supported(operation->output_tensors[0]);
            break;
         }
         supported = getenv("ETNA_AVGPOOL") && is_global_avg_pool(operation) &&
                     tensor_quantization_supported(operation->input_tensors[0]) &&
                     tensor_quantization_supported(operation->output_tensors[0]);
         break;
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
         /* Local experiment ETNA_NO_LOGISTIC: leave LOGISTIC to the CPU (in SSD heads it makes a 2-op subgraph) */
         supported = !getenv("ETNA_NO_LOGISTIC");
         break;
      case PIPE_ML_OPERATION_TYPE_RESHAPE:
      case PIPE_ML_OPERATION_TYPE_RELU:
      case PIPE_ML_OPERATION_TYPE_ABSOLUTE:
         supported = true;
         break;
      default:
         return false;
   }

   return supported;
}


/* Local experiment ETNA_SRAM_ACT=<bytes>: keep activations that only flow between NN jobs in VIP SRAM,
 * like the blob. GL_SRAM_REMAP_ADDRESS maps the SRAM over a reserved range of the NPU address space.
 * Tensors get SRAM space first-fit over their live range; the caches of every job that runs while
 * an SRAM tensor is live are moved above it. */
static void
plan_sram(struct etna_ml_subgraph *subgraph, struct list_head *etna_operations)
{
   const char *env = getenv("ETNA_SRAM_ACT");
   if (!env || subgraph->screen->specs.nn_core_version != 8)
      return;

   unsigned budget = strtoul(env, NULL, 0);
   unsigned base = 0x800;
   unsigned n = list_length(etna_operations);
   unsigned tensor_count = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
   struct etna_operation **ops = calloc(n, sizeof(*ops));
   struct { unsigned start, end, first, last; } *allocs = calloc(tensor_count, sizeof(*allocs));
   bool *done = calloc(tensor_count, sizeof(*done));
   unsigned nalloc = 0, k = 0;

   list_for_each_entry(struct etna_operation, op, etna_operations, link)
      ops[k++] = op;

   /* Plan per buffer: an addition reads both of its inputs from one buffer, so they move together.
    * A buffer qualifies if only NN jobs touch its tensors and none of them is a graph input or output. */
   for (unsigned idx = 0; idx < tensor_count; idx++) {
      struct etna_ml_tensor *t = *util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, idx);
      if (!t || !t->resource || done[idx])
         continue;

      struct pipe_resource *res = t->resource;
      unsigned first = n, last = 0;
      bool ok = true;

      for (unsigned j = 0; j < tensor_count; j++) {
         struct etna_ml_tensor *o = *util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, j);
         if (!o || o->resource != res)
            continue;
         done[j] = true;

         bool produced = false, consumed = false;
         for (unsigned c = 0; c < n; c++) {
            bool job = ops[c]->type == ETNA_JOB_TYPE_NN || ops[c]->type == ETNA_JOB_TYPE_TP;
            bool touches = false;
            for (unsigned i = 0; i < ops[c]->output_count; i++)
               if (ops[c]->output_tensors[i] == j)
                  produced = touches = true;
            for (unsigned i = 0; i < ops[c]->input_count; i++)
               if (ops[c]->input_tensors[i] == j)
                  consumed = touches = true;
            if (touches && job) {
               first = MIN2(first, c);
               last = MAX2(last, c);
               if (ops[c]->type != ETNA_JOB_TYPE_NN)
                  ok = false;
            }
         }
         if (!produced || !consumed)
            ok = false;
      }

      unsigned size = align(pipe_buffer_size(res), 256);
      if (!ok || first >= n || size > budget)
         continue;

      const char *maxb = getenv("ETNA_SRAM_MAX");
      if (maxb && nalloc >= strtoul(maxb, NULL, 0))
         continue;

      unsigned off = base;
      bool moved = true;
      while (moved) {
         moved = false;
         for (unsigned a = 0; a < nalloc; a++) {
            bool live = !(last < allocs[a].first || first > allocs[a].last);
            if (live && off < allocs[a].end && off + size > allocs[a].start) {
               off = allocs[a].end;
               moved = true;
            }
         }
      }
      if (off + size > base + budget)
         continue;

      allocs[nalloc].start = off;
      allocs[nalloc].end = off + size;
      allocs[nalloc].first = first;
      allocs[nalloc].last = last;
      nalloc++;

      for (unsigned j = 0; j < tensor_count; j++) {
         struct etna_ml_tensor *o = *util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, j);
         if (o && o->resource == res) {
            o->in_sram = true;
            o->sram_offset = off + o->offset;
         }
      }
      ML_DBG("SRAMACT buffer of tensor %u size %u at 0x%x live %u-%u\n", idx, size, off, first, last);
      for (unsigned j = 0; j < tensor_count; j++) {
         struct etna_ml_tensor *o = *util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, j);
         if (o && o->resource == res)
            ML_DBG("SRAMACT   tensor %u offset %u size %u\n", j, o->offset, o->size);
      }
   }

   if (nalloc > 0) {
      for (k = 0; k < n; k++) {
         unsigned top = 0;
         for (unsigned a = 0; a < nalloc; a++)
            if (allocs[a].first <= k && k <= allocs[a].last)
               top = MAX2(top, allocs[a].end);
         ops[k]->sram_cache_base = top;
      }

      const char *al = getenv("ETNA_SRAM_ALIGN");
      unsigned window_align = al ? strtoul(al, NULL, 0) : 0x40000;
      unsigned oc_sram_size = etna_ml_get_core_info(subgraph->screen)->on_chip_sram_size;
      subgraph->sram_bo = etna_ml_create_bo(subgraph->screen, oc_sram_size + window_align);
      subgraph->sram_window = align(etna_bo_gpu_va(subgraph->sram_bo), window_align);
      ML_DBG("SRAMACT window 0x%x, %u buffers\n", subgraph->sram_window, nalloc);
   }

   free(done);
   free(allocs);
   free(ops);
}

struct pipe_ml_subgraph *
etna_ml_subgraph_create(struct pipe_ml_device *pdevice,
                        const struct pipe_ml_operation *poperations,
                        unsigned count)
{
   struct etna_screen *screen = etna_ml_device_screen(pdevice);
   unsigned nn_core_count = etna_ml_get_core_info(screen)->nn_core_count;
   struct etna_ml_subgraph *subgraph;
   struct list_head operations;
   unsigned tensor_count;

   if (nn_core_count < 1) {
      fprintf(stderr, "We need at least 1 NN core to do anything useful.\n");
      abort();
   }

   subgraph = calloc(1, sizeof(*subgraph));
   tensor_count = count_tensors(poperations, count);

   list_inithead(&operations);

   subgraph->base.device = pdevice;
   subgraph->screen = screen;
   subgraph->operations = UTIL_DYNARRAY_INIT;

   subgraph->tensors = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_resize(&subgraph->tensors, struct etna_ml_tensor*, tensor_count))
      return NULL;
   memset(util_dynarray_begin(&subgraph->tensors), 0, subgraph->tensors.size);

   lower_operations(subgraph, poperations, count, &operations);
   plan_sram(subgraph, &operations);

   list_for_each_entry(struct etna_operation, operation, &operations, link) {
      struct etna_vip_instruction instruction = {0};

      if (operation->cpu_transpose)
         continue;

      switch(operation->type) {
         case ETNA_JOB_TYPE_NN:
            etna_ml_compile_operation_nn(subgraph, operation, &instruction);
            break;
         case ETNA_JOB_TYPE_TP:
            etna_ml_compile_operation_tp(subgraph, operation, &instruction);
            break;
         case ETNA_JOB_TYPE_CONCAT:
         case ETNA_JOB_TYPE_SPLIT:
         case ETNA_JOB_TYPE_BYPASS:
            continue;
      }

      util_dynarray_append(&subgraph->operations, instruction);
   }

   /* Local experiment: buffers holding a graph input (no producer) or output (no consumer) */
   {
      unsigned tensor_count = util_dynarray_num_elements(&subgraph->tensors, struct etna_ml_tensor *);
      struct util_dynarray ext = UTIL_DYNARRAY_INIT;
      for (unsigned j = 0; j < tensor_count; j++) {
         struct etna_ml_tensor *t = *util_dynarray_element(&subgraph->tensors, struct etna_ml_tensor *, j);
         if (!t || !t->resource)
            continue;
         bool produced = false, consumed = false;
         list_for_each_entry(struct etna_operation, op, &operations, link) {
            for (unsigned i = 0; i < op->output_count; i++)
               produced |= op->output_tensors[i] == j;
            for (unsigned i = 0; i < op->input_count; i++)
               consumed |= op->input_tensors[i] == j;
         }
         if (!produced || !consumed) {
            util_dynarray_append(&ext, t->resource);
            ML_DBG("EXTERNAL tensor %u produced %d consumed %d size %u offset %u\n", j, produced, consumed, t->size, t->offset);
         }
      }
      util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, inst) {
         util_dynarray_foreach(&ext, struct pipe_resource *, r) {
            inst->input_external |= inst->input == *r;
            inst->output_external |= inst->output == *r;
         }
      }
      util_dynarray_fini(&ext);
   }

   list_for_each_entry_safe(struct etna_operation, operation, &operations, link) {
      free(operation->weight_tensor);
      free(operation->bias_tensor);
      free(operation);
   }

   return &subgraph->base;
}

static void
dump_buffer(const uint8_t *ptr, char *name, int operation_nr, int suboperation_nr, int offset, unsigned size)
{
   char buffer[255];

   snprintf(buffer, sizeof(buffer), "mesa-%s-%03u-%03u.bin", name, operation_nr, suboperation_nr);

   ML_DBG("Dumping buffer from 0x%lx at offset %d with size %d to %s\n", ptr, offset, size, buffer);

   FILE *f = fopen(buffer, "wb");
   assert(f);
   fwrite(ptr + offset, 1, size, f);
   if(ferror(f)) {
      ML_DBG("Error in writing to file: %s\n", strerror(errno));
   }
   fflush(f);
   fclose(f);
}

static void
dump_bo(struct etna_bo *bo, char *name, int operation_nr, int suboperation_nr, int offset, int size)
{
   const uint8_t *map = etna_bo_map(bo);
   if (size == 0)
      size = etna_bo_size(bo) - offset;
   dump_buffer(map, name, operation_nr, suboperation_nr, offset, size);
}

static void
init_npu(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_cmd_stream *stream = ctx->stream;

   /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   etna_set_state(stream, VIVS_PA_SYSTEM_MODE, VIVS_PA_SYSTEM_MODE_PROVOKING_VERTEX_LAST |
                                               VIVS_PA_SYSTEM_MODE_HALF_PIXEL_CENTER);
   etna_set_state(stream, VIVS_GL_API_MODE, VIVS_GL_API_MODE_OPENCL);

   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   pctx->flush(pctx, NULL, 0);
}

static void
close_batch(struct pipe_context *pctx)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_cmd_stream *stream = ctx->stream;

   unsigned cache = VIVS_GL_FLUSH_CACHE_DEPTH | VIVS_GL_FLUSH_CACHE_COLOR | VIVS_GL_FLUSH_CACHE_UNK10;
   if (!DBG_ENABLED(ETNA_DBG_NPU_PARALLEL))
      cache |= VIVS_GL_FLUSH_CACHE_UNK11 | VIVS_GL_FLUSH_CACHE_SHADER_L1;

   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, cache);
   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, cache);

   etna_cmd_stream_emit(stream, 0x0);
   etna_cmd_stream_emit(stream, 0x0);

   ctx->dirty = 0;
}


/* Local experiment: per-inference phase timing (ETNA_ML_TIMING=1) */
static long mlt_t[6];
static long
mlt_now(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1000000L + t.tv_nsec / 1000L;
}
#define MLT(i) do { if (ETNA_ML_ENV("ETNA_ML_TIMING")) mlt_t[i] = mlt_now(); } while (0)

/* Local experiment: CPU writes into tensor BOs with explicit cache maintenance. The buffer transfer path maps
 * never-written ranges UNSYNCHRONIZED and then skips etna_bo_cpu_fini, which leaves a cached BO dirty. */
static uint8_t *
ml_map_write(struct pipe_resource *res, unsigned offset)
{
   struct etna_bo *bo = etna_buffer_resource(res)->bo;
   etna_bo_cpu_prep(bo, DRM_ETNA_PREP_WRITE);
   return (uint8_t *)etna_bo_map(bo) + offset;
}

static void
ml_unmap_write(struct pipe_resource *res)
{
   etna_bo_cpu_fini(etna_buffer_resource(res)->bo);
}

static void
ml_dump(const char *kind, unsigned idx, const void *data, unsigned size)
{
   const char *dir = getenv("ETNA_ML_DUMPIO");
   if (!dir)
      return;
   char path[256];
   snprintf(path, sizeof(path), "%s/%s_%u.bin", dir, kind, idx);
   FILE *f = fopen(path, "wb");
   if (f) {
      fwrite(data, 1, size, f);
      fclose(f);
   }
}

/* Local experiment ETNA_ML_THREADS=n: run fn(arg, part, parts) on the caller and n - 1 workers */
typedef void (*ml_par_fn)(void *arg, unsigned part, unsigned parts);

struct ml_par_job {
   ml_par_fn fn;
   void *arg;
   unsigned part, parts;
   struct util_queue_fence fence;
};

static struct util_queue ml_queue;
static unsigned ml_threads;

static void
ml_par_execute(void *data, void *gdata, int thread_index)
{
   struct ml_par_job *job = data;
   job->fn(job->arg, job->part, job->parts);
}

static void
ml_parallel(ml_par_fn fn, void *arg)
{
   if (!ml_threads) {
      const char *env = getenv("ETNA_ML_THREADS");
      ml_threads = CLAMP(env ? atoi(env) : 1, 1, 4);
      if (ml_threads > 1 && !util_queue_init(&ml_queue, "etna_ml", 8, ml_threads - 1, 0, NULL))
         ml_threads = 1;
   }

   struct ml_par_job jobs[4];
   for (unsigned i = 1; i < ml_threads; i++) {
      jobs[i] = (struct ml_par_job){ .fn = fn, .arg = arg, .part = i, .parts = ml_threads };
      util_queue_fence_init(&jobs[i].fence);
      util_queue_add_job(&ml_queue, &jobs[i], &jobs[i].fence, ml_par_execute, NULL, 0);
   }
   fn(arg, 0, ml_threads);
   for (unsigned i = 1; i < ml_threads; i++) {
      util_queue_fence_wait(&jobs[i].fence);
      util_queue_fence_destroy(&jobs[i].fence);
   }
}

/* Local experiment: write an NHWC graph input as planar (out[c * h * w + p] = in[p * c + ch]), as the
 * TP transpose job would, adding 128 for signed inputs. */
static void
cpu_transpose_input(struct pipe_context *pctx, struct etna_ml_subgraph *subgraph,
                    const struct etna_cpu_transpose *ct, const uint8_t *src, bool is_signed)
{
   struct etna_ml_tensor *out = etna_ml_get_tensor(subgraph, ct->out_idx);
   unsigned pixels = ct->h * ct->w;
   struct pipe_transfer *transfer;
   long tt0 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;
   uint8_t *dst = ml_map_write(out->resource, out->offset);
   (void)transfer;
   uint8_t flip = is_signed ? 0x80 : 0x0;
   unsigned p = 0;
   long tt1 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;

#if defined(__aarch64__)
   if (ct->c == 3) {
      uint8x16_t vflip = vdupq_n_u8(flip);
      for (; p + 16 <= pixels; p += 16) {
         uint8x16x3_t v = vld3q_u8(src + 3 * p);
         vst1q_u8(dst + p, veorq_u8(v.val[0], vflip));
         vst1q_u8(dst + pixels + p, veorq_u8(v.val[1], vflip));
         vst1q_u8(dst + 2 * pixels + p, veorq_u8(v.val[2], vflip));
      }
   }
#endif
   for (; p < pixels; p++)
      for (unsigned c = 0; c < ct->c; c++)
         dst[c * pixels + p] = src[p * ct->c + c] ^ flip;

   long tt2 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;
   ml_unmap_write(out->resource);
   if (ETNA_ML_ENV("ETNA_ML_TIMING"))
      fprintf(stderr, "CPUTRANSPOSE bytes %u map %ld loop %ld unmap %ld\n", pixels * ct->c, tt1 - tt0, tt2 - tt1, mlt_now() - tt2);
}

/* Local experiment: NHWC graph input straight to the stride-2 destrided layout of the TP reshuffle:
 * out[(c * 4 + py * 2 + px) * P + y2 * out_w + x2] = in(h = 2 * y2 + py - pad_y, w = 2 * x2 + px - pad_x), else zp */
struct reshuffle_rows_args {
   const struct etna_cpu_transpose *ct;
   const uint8_t *src;
   uint8_t *dst;
   uint8_t flip;
};

/* Input rows [part * H / parts, (part + 1) * H / parts): each lands in its own output rows */
static void
reshuffle_rows_part(void *arg, unsigned part, unsigned parts)
{
   const struct reshuffle_rows_args *a = arg;
   const struct etna_cpu_transpose *ct = a->ct;
   const uint8_t *src = a->src;
   uint8_t *dst = a->dst;
   uint8_t flip = a->flip;
   unsigned P = ct->out_w * ct->out_h, W = ct->w, H = ct->h, C = ct->c;
   unsigned h0 = H * part / parts, h1 = H * (part + 1) / parts;

   for (unsigned h = h0; h < h1; h++) {
      unsigned y = h + ct->pad_y, py = y & 1, y2 = y >> 1;
      const uint8_t *in = src + h * W * C;
      unsigned w = 0;
#if defined(__aarch64__)
      if (C == 3 && (ct->pad_x & 1) == 0) {
         /* even pad_x keeps the column phase: x = w + pad_x, x2 = w / 2 + pad_x / 2 */
         uint8x16_t vflip = vdupq_n_u8(flip);
         uint8_t *d[3][2];
         for (unsigned c = 0; c < 3; c++)
            for (unsigned px = 0; px < 2; px++)
               d[c][px] = dst + (c * 4 + py * 2 + px) * P + y2 * ct->out_w + ct->pad_x / 2;
         for (; w + 32 <= W; w += 32) {
            uint8x16x3_t a = vld3q_u8(in + 3 * w);
            uint8x16x3_t b = vld3q_u8(in + 3 * (w + 16));
            for (unsigned c = 0; c < 3; c++) {
               vst1q_u8(d[c][0] + w / 2, veorq_u8(vuzp1q_u8(a.val[c], b.val[c]), vflip));
               vst1q_u8(d[c][1] + w / 2, veorq_u8(vuzp2q_u8(a.val[c], b.val[c]), vflip));
            }
         }
      }
#endif
      for (; w < W; w++) {
         unsigned x = w + ct->pad_x;
         for (unsigned c = 0; c < C; c++)
            dst[(c * 4 + py * 2 + (x & 1)) * P + y2 * ct->out_w + (x >> 1)] = in[w * C + c] ^ flip;
      }
   }

}

static void
cpu_reshuffle_input(struct pipe_context *pctx, struct etna_ml_subgraph *subgraph,
                    const struct etna_cpu_transpose *ct, const uint8_t *src, bool is_signed)
{
   struct etna_ml_tensor *out = etna_ml_get_tensor(subgraph, ct->out_idx);
   unsigned P = ct->out_w * ct->out_h, W = ct->w, H = ct->h, C = ct->c;
   struct pipe_transfer *transfer;
   long tt0 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;
   uint8_t *dst = ml_map_write(out->resource, out->offset);
   (void)transfer;
   uint8_t flip = is_signed ? 0x80 : 0x0;
   long tt1 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;

   /* Output positions that map outside the input take the zero point */
   for (unsigned c = 0; c < C; c++) {
      for (unsigned py = 0; py < 2; py++) {
         for (unsigned px = 0; px < 2; px++) {
            uint8_t *plane = dst + (c * 4 + py * 2 + px) * P;
            /* valid x2: 0 <= 2 * x2 + px - pad_x < W */
            unsigned x_lo = ct->pad_x > px ? (ct->pad_x - px + 1) / 2 : 0;
            unsigned x_hi = MIN2((W - 1 + ct->pad_x - px) / 2 + 1, ct->out_w);
            for (unsigned y2 = 0; y2 < ct->out_h; y2++) {
               int h = 2 * y2 + py - ct->pad_y;
               uint8_t *row = plane + y2 * ct->out_w;
               if (h < 0 || h >= (int)H) {
                  memset(row, ct->zp, ct->out_w);
                  continue;
               }
               if (x_lo > 0)
                  memset(row, ct->zp, x_lo);
               if (x_hi < ct->out_w)
                  memset(row + x_hi, ct->zp, ct->out_w - x_hi);
            }
         }
      }
   }

   struct reshuffle_rows_args args = { ct, src, dst, flip };
   ml_parallel(reshuffle_rows_part, &args);

   long tt2 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;
   ml_unmap_write(out->resource);
   if (ETNA_ML_ENV("ETNA_ML_TIMING"))
      fprintf(stderr, "CPURESHUFFLE bytes %u map %ld loop %ld unmap %ld\n", P * C * 4, tt1 - tt0, tt2 - tt1, mlt_now() - tt2);
}

void
etna_ml_subgraph_invoke(struct pipe_context *pctx, struct pipe_ml_subgraph *psubgraph,
                        unsigned inputs_count, unsigned input_idxs[], void *inputs[],
                        bool is_signed[])
{
   struct etna_context *ctx = etna_context(pctx);
   unsigned tp_core_count = etna_ml_get_core_info(ctx->screen)->tp_core_count;
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);
   struct etna_cmd_stream *stream = ctx->stream;
   static bool is_initialized = false;

   if (!is_initialized) {
      init_npu(pctx);
      is_initialized = true;
   }

   MLT(0);

   if (!DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
      /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
      etna_cmd_stream_emit(stream, 0x0);
   }

   for (int i = 0; i < inputs_count; i++) {
      struct etna_ml_tensor *tensor = etna_ml_get_tensor(subgraph, input_idxs[i]);
      bool done_on_cpu = false;
      util_dynarray_foreach(&subgraph->cpu_transposes, struct etna_cpu_transpose, ct) {
         if (ct->in_idx == input_idxs[i]) {
            if (ct->stride)
               cpu_reshuffle_input(pctx, subgraph, ct, inputs[i], is_signed[i]);
            else
               cpu_transpose_input(pctx, subgraph, ct, inputs[i], is_signed[i]);
            done_on_cpu = true;
         }
      }
      ml_dump("in", input_idxs[i], inputs[i], tensor->size);
      if (done_on_cpu)
         continue;
      uint8_t *dst_map = ml_map_write(tensor->resource, tensor->offset);
      if (is_signed[i]) {
         const uint8_t *src = inputs[i];
         for (unsigned k = 0; k < tensor->size; k++)
            dst_map[k] = src[k] + 128;
      } else {
         memcpy(dst_map, inputs[i], tensor->size);
      }
      ml_unmap_write(tensor->resource);
   }

   if (subgraph->sram_bo)
      etna_cmd_stream_ref_bo(stream, subgraph->sram_bo, ETNA_RELOC_READ | ETNA_RELOC_WRITE);

   MLT(1);

   subgraph->prefix_limit = -1;
   if (ETNA_ML_ENV("ETNA_PREFIX")) {
      unsigned n = util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction);
      subgraph->prefix_limit = subgraph->prefix_iter++ % (n + 1);
   }

   bool replay = ETNA_ML_ENV("ETNA_ML_CMD_CACHE") && ETNA_ML_ENV("ETNA_ML_LIST_ONCE") && subgraph->listed_once &&
                 subgraph->prefix_limit < 0 && !DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING) &&
                 !DBG_ENABLED(ETNA_DBG_DUMP_SHADERS);
   uint32_t *record_base = stream->buffer;
   uint32_t record_start = stream->offset;

   if (replay && subgraph->cmd_cache) {
      util_dynarray_foreach(&subgraph->cmd_bos, struct etna_bo *, bo)
         etna_cmd_stream_ref_bo(stream, *bo, ETNA_RELOC_READ | ETNA_RELOC_WRITE);
      etna_cmd_stream_reserve(stream, subgraph->cmd_cache_words);
      memcpy(stream->buffer + stream->offset, subgraph->cmd_cache, subgraph->cmd_cache_words * 4);
      stream->offset += subgraph->cmd_cache_words;
      goto jobs_done;
   }

   static unsigned i = 0;
   util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, operation) {
      if (subgraph->prefix_limit >= 0 &&
          operation - (struct etna_vip_instruction *)util_dynarray_begin(&subgraph->operations) >= subgraph->prefix_limit)
         break;

      if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS)) {
         struct pipe_transfer *transfer = NULL;
         pipe_buffer_map(pctx, operation->input, PIPE_MAP_READ, &transfer);
         dump_bo(etna_buffer_resource(operation->input)->bo, "input", i, 0, operation->input_offset, 0);
         pipe_buffer_unmap(pctx, transfer);

         switch (operation->type) {
            case ETNA_JOB_TYPE_TP:
               for (unsigned j = 0; j < tp_core_count && operation->configs[j]; j++) {
                  dump_bo(operation->configs[j], "tp", i, j, 0, 0);
                  if (operation->pwl_lut)
                     dump_bo(operation->pwl_lut, "lut", i, j, 0, 0);
               }
               break;
            case ETNA_JOB_TYPE_NN:
               dump_bo(operation->configs[0], "nn", i, 0, 0, 0);
               dump_bo(operation->coefficients, "compressed", i, 0, 0, 0);
               break;
            default:
               UNREACHABLE("Unsupported ML operation type");
         }
      }

      if (DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
         /* These zeroes match the blob's cmdstream. They are here to make diff'ing easier.*/
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
         etna_cmd_stream_emit(stream, 0x0);
      }

      bool list_consts = !ETNA_ML_ENV("ETNA_ML_LIST_ONCE") || !subgraph->listed_once;
      for (unsigned j = 0; list_consts && j < tp_core_count && operation->configs[j]; j++)
         etna_cmd_stream_ref_bo(stream, operation->configs[j], ETNA_RELOC_READ);
      if (list_consts && operation->coefficients)
         etna_cmd_stream_ref_bo(stream, operation->coefficients, ETNA_RELOC_READ);
      if (list_consts && operation->pwl_lut)
         etna_cmd_stream_ref_bo(stream, operation->pwl_lut, ETNA_RELOC_READ);
      unsigned op_idx = operation - (struct etna_vip_instruction *)util_dynarray_begin(&subgraph->operations);
      bool waited_on = subgraph->prefix_limit >= 0 ? op_idx + 1 == (unsigned)subgraph->prefix_limit
                                                   : op_idx + 1 == util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction);
      if (list_consts || operation->input_external)
         etna_cmd_stream_ref_bo(stream, etna_buffer_resource(operation->input)->bo, ETNA_RELOC_READ);
      if (list_consts || operation->output_external || waited_on)
         etna_cmd_stream_ref_bo(stream, etna_buffer_resource(operation->output)->bo, ETNA_RELOC_WRITE);

      switch (operation->type) {
         case ETNA_JOB_TYPE_TP:
            etna_ml_emit_operation_tp(pctx, subgraph, operation, i);
            break;
         case ETNA_JOB_TYPE_NN:
            etna_ml_emit_operation_nn(pctx, subgraph, operation, i);
            break;
         default:
            UNREACHABLE("Unsupported ML operation type");
      }

      if (DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING)) {
         ML_DBG("Running operation %d - %d\n", i, operation->type);
         close_batch(pctx);

         if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS))
            dump_buffer((uint8_t *)ctx->stream->buffer, "cmd", i, 0, 0, ctx->stream->offset * 4);

         /* Local profiling hack: wait for each job and print its wall time in us. */
         struct timespec op_t0, op_t1;
         clock_gettime(CLOCK_MONOTONIC, &op_t0);
         pctx->flush(pctx, NULL, 0);
         {
            struct pipe_transfer *wait_transfer = NULL;
            pipe_buffer_map(pctx, operation->output, PIPE_MAP_READ, &wait_transfer);
            pipe_buffer_unmap(pctx, wait_transfer);
         }
         clock_gettime(CLOCK_MONOTONIC, &op_t1);
         ML_DBG("OPTIME op %d type %s out_bytes %u us %ld\n", i,
                operation->type == ETNA_JOB_TYPE_NN ? "NN" : "TP",
                (unsigned)pipe_buffer_size(operation->output),
                (op_t1.tv_sec - op_t0.tv_sec) * 1000000L + (op_t1.tv_nsec - op_t0.tv_nsec) / 1000L);

         if (DBG_ENABLED(ETNA_DBG_DUMP_SHADERS)) {
            struct pipe_transfer *transfer = NULL;

            pipe_buffer_map(pctx, operation->output, PIPE_MAP_READ, &transfer);
            dump_bo(etna_buffer_resource(operation->output)->bo, "output", i, 0, operation->output_offset, 0);
            pipe_buffer_unmap(pctx, transfer);
         }

         stream = ctx->stream;
      }

      i++;
   }

   if (replay && !subgraph->cmd_cache && stream->buffer == record_base && stream->offset > record_start) {
      unsigned n = util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction);
      subgraph->cmd_cache_words = stream->offset - record_start;
      subgraph->cmd_cache = malloc(subgraph->cmd_cache_words * 4);
      memcpy(subgraph->cmd_cache, stream->buffer + record_start, subgraph->cmd_cache_words * 4);
      util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, inst) {
         unsigned idx = inst - (struct etna_vip_instruction *)util_dynarray_begin(&subgraph->operations);
         if (inst->input_external)
            util_dynarray_append(&subgraph->cmd_bos, etna_buffer_resource(inst->input)->bo);
         if (inst->output_external || idx + 1 == n)
            util_dynarray_append(&subgraph->cmd_bos, etna_buffer_resource(inst->output)->bo);
      }
      ML_DBG("CMDCACHE recorded %u words, %u BOs\n", subgraph->cmd_cache_words,
             util_dynarray_num_elements(&subgraph->cmd_bos, struct etna_bo *));
   }

jobs_done:
   if (!DBG_ENABLED(ETNA_DBG_NPU_NO_BATCHING))
      close_batch(pctx);

   if (subgraph->prefix_limit < 0 ||
       (unsigned)subgraph->prefix_limit == util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction))
      subgraph->listed_once = true;
   MLT(2);

   if (DBG_ENABLED(ETNA_DBG_FLUSH_ALL))
      pctx->flush(pctx, NULL, 0);
}

#if defined(__aarch64__)
/* rows: 8 channel planes (pstride apart) of 8 pixels; written as 8 pixels of 8 channels (ostride apart) */
static inline void
detranspose_8x8(const uint8_t *in, unsigned pstride, uint8_t *out, unsigned ostride, uint8x8_t vflip)
{
   uint8x8_t r[8];
   for (unsigned i = 0; i < 8; i++)
      r[i] = veor_u8(vld1_u8(in + i * pstride), vflip);

   uint8x8x2_t t01 = vtrn_u8(r[0], r[1]), t23 = vtrn_u8(r[2], r[3]);
   uint8x8x2_t t45 = vtrn_u8(r[4], r[5]), t67 = vtrn_u8(r[6], r[7]);
   uint16x4x2_t u02 = vtrn_u16(vreinterpret_u16_u8(t01.val[0]), vreinterpret_u16_u8(t23.val[0]));
   uint16x4x2_t u13 = vtrn_u16(vreinterpret_u16_u8(t01.val[1]), vreinterpret_u16_u8(t23.val[1]));
   uint16x4x2_t u46 = vtrn_u16(vreinterpret_u16_u8(t45.val[0]), vreinterpret_u16_u8(t67.val[0]));
   uint16x4x2_t u57 = vtrn_u16(vreinterpret_u16_u8(t45.val[1]), vreinterpret_u16_u8(t67.val[1]));
   uint32x2x2_t v04 = vtrn_u32(vreinterpret_u32_u16(u02.val[0]), vreinterpret_u32_u16(u46.val[0]));
   uint32x2x2_t v15 = vtrn_u32(vreinterpret_u32_u16(u13.val[0]), vreinterpret_u32_u16(u57.val[0]));
   uint32x2x2_t v26 = vtrn_u32(vreinterpret_u32_u16(u02.val[1]), vreinterpret_u32_u16(u46.val[1]));
   uint32x2x2_t v37 = vtrn_u32(vreinterpret_u32_u16(u13.val[1]), vreinterpret_u32_u16(u57.val[1]));

   vst1_u8(out + 0 * ostride, vreinterpret_u8_u32(v04.val[0]));
   vst1_u8(out + 1 * ostride, vreinterpret_u8_u32(v15.val[0]));
   vst1_u8(out + 2 * ostride, vreinterpret_u8_u32(v26.val[0]));
   vst1_u8(out + 3 * ostride, vreinterpret_u8_u32(v37.val[0]));
   vst1_u8(out + 4 * ostride, vreinterpret_u8_u32(v04.val[1]));
   vst1_u8(out + 5 * ostride, vreinterpret_u8_u32(v15.val[1]));
   vst1_u8(out + 6 * ostride, vreinterpret_u8_u32(v26.val[1]));
   vst1_u8(out + 7 * ostride, vreinterpret_u8_u32(v37.val[1]));
}
#endif

static void
detranspose_range(const uint8_t *src, uint8_t *dst, unsigned pixels, unsigned channels,
                  unsigned p0, unsigned p1, uint8_t flip)
{
   unsigned p = p0;
#if defined(__aarch64__)
   uint8x8_t vflip = vdup_n_u8(flip);
   for (; p + 8 <= p1; p += 8) {
      unsigned c = 0;
      for (; c + 8 <= channels; c += 8)
         detranspose_8x8(src + c * pixels + p, pixels, dst + p * channels + c, channels, vflip);
      for (; c < channels; c++)
         for (unsigned j = 0; j < 8; j++)
            dst[(p + j) * channels + c] = src[c * pixels + p + j] ^ flip;
   }
#endif
   for (; p < p1; p++) {
      const uint8_t *in = src + p;
      uint8_t *out = dst + p * channels;
      for (unsigned c = 0; c < channels; c++)
         out[c] = in[c * pixels] ^ flip;
   }
}

struct ml_detranspose_task {
   const struct etna_cpu_detranspose *cd;
   const uint8_t *src;
   uint8_t *dst;
   uint8_t flip;
   struct pipe_transfer *transfer;
};

struct ml_detranspose_tasks {
   struct ml_detranspose_task *task;
   unsigned count;
};

/* Every thread converts a slice of 8-pixel blocks of every output */
static void
detranspose_part(void *arg, unsigned part, unsigned parts)
{
   const struct ml_detranspose_tasks *tasks = arg;
   for (unsigned i = 0; i < tasks->count; i++) {
      const struct ml_detranspose_task *t = &tasks->task[i];
      unsigned pixels = t->cd->pixels;
      unsigned chunk = align(DIV_ROUND_UP(pixels, parts), 8);
      unsigned p0 = MIN2(part * chunk, pixels), p1 = MIN2(p0 + chunk, pixels);
      detranspose_range(t->src, t->dst, pixels, t->cd->c, p0, p1, t->flip);
   }
}


void
etna_ml_subgraph_read_outputs(struct pipe_context *context, struct pipe_ml_subgraph *psubgraph,
                              unsigned outputs_count, unsigned output_idxs[], void *outputs[],
                              bool is_signed[])
{
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);
   unsigned operation_count = util_dynarray_num_elements(&subgraph->operations, struct etna_vip_instruction);
   struct etna_vip_instruction *last_operation;

   /* Local experiment ETNA_PREFIX: wait for the last job that was actually submitted */
   if (subgraph->prefix_limit >= 0)
      operation_count = MAX2(subgraph->prefix_limit, 1);

   if (operation_count > 0) {
      last_operation = util_dynarray_element(&subgraph->operations,
                                             struct etna_vip_instruction,
                                             operation_count - 1);

      if (DBG_ENABLED(ETNA_DBG_ML_MSGS)) {
         long start, end;
         struct timespec time;

         clock_gettime(CLOCK_MONOTONIC, &time);
         start = (long)time.tv_sec * 1000 + (long)time.tv_nsec / 1000000;

         context->flush(context, NULL, 0);

         struct pipe_transfer *transfer = NULL;
         pipe_buffer_map(context, last_operation->output, PIPE_MAP_READ, &transfer);
         pipe_buffer_unmap(context, transfer);

         clock_gettime(CLOCK_MONOTONIC, &time);
         end = (long)time.tv_sec * 1000 + (long)time.tv_nsec / 1000000;
         ML_DBG("Running the NN job took %ld ms.\n", (end - start));
      } else {
         MLT(3);
         context->flush(context, NULL, 0);
         MLT(4);
         if (ETNA_ML_ENV("ETNA_ML_TIMING")) {
            struct pipe_transfer *transfer = NULL;
            pipe_buffer_map(context, last_operation->output, PIPE_MAP_READ, &transfer);
            pipe_buffer_unmap(context, transfer);
         }
      }
   }
   MLT(5);

   /* Local experiment ETNA_CPU_DETRANSPOSE: all planar outputs in one go, split over ETNA_ML_THREADS */
   unsigned cd_count = util_dynarray_num_elements(&subgraph->cpu_detransposes, struct etna_cpu_detranspose);
   if (cd_count) {
      struct ml_detranspose_task *task = calloc(cd_count, sizeof(*task));
      unsigned n = 0;
      for (int i = 0; i < outputs_count; i++) {
         util_dynarray_foreach(&subgraph->cpu_detransposes, struct etna_cpu_detranspose, cd) {
            if (cd->idx != output_idxs[i] || n == cd_count)
               continue;
            struct etna_ml_tensor *t = etna_ml_get_tensor(subgraph, cd->idx);
            task[n].cd = cd;
            task[n].dst = outputs[i];
            task[n].flip = is_signed[i] ? 0x80 : 0x0;
            task[n].src = pipe_buffer_map_range(context, t->resource, t->offset, cd->pixels * cd->c,
                                                PIPE_MAP_READ, &task[n].transfer);
            n++;
         }
      }
      struct ml_detranspose_tasks tasks = { task, n };
      ml_parallel(detranspose_part, &tasks);
      for (unsigned k = 0; k < n; k++)
         pipe_buffer_unmap(context, task[k].transfer);
      free(task);
   }

   for (int i = 0; i < outputs_count; i++) {
      struct pipe_resource *res = etna_ml_get_resource(subgraph, output_idxs[i]);
      long out_t0 = ETNA_ML_ENV("ETNA_ML_TIMING") ? mlt_now() : 0;
      const struct etna_cpu_detranspose *cpu_detranspose = NULL;
      util_dynarray_foreach(&subgraph->cpu_detransposes, struct etna_cpu_detranspose, cd)
         if (cd->idx == output_idxs[i])
            cpu_detranspose = cd;
      if (cpu_detranspose) {
         /* converted above */
      } else if (is_signed[i]) {
         struct pipe_transfer *src_transfer;
         uint8_t *src_map;
         src_map = (uint8_t *) pipe_buffer_map_range(context,
                                                     res,
                                                     0, pipe_buffer_size(res),
                                                     PIPE_MAP_READ,
                                                     &src_transfer);
         assert(src_map);
         for (unsigned k = 0; k < etna_ml_get_size(subgraph, output_idxs[i]); k++) {
            ((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;
         }
         pipe_buffer_unmap(context, src_transfer);
      } else {
         pipe_buffer_read(context, res, 0, etna_ml_get_size(subgraph, output_idxs[i]), outputs[i]);
      }
      ml_dump("out", output_idxs[i], outputs[i], etna_ml_get_size(subgraph, output_idxs[i]));
      if (ETNA_ML_ENV("ETNA_ML_TIMING"))
         fprintf(stderr, "MLOUT %d tensor %u signed %d bytes %u bo %u cached %d us %ld\n", i, output_idxs[i], is_signed[i],
                 etna_ml_get_size(subgraph, output_idxs[i]), (unsigned)pipe_buffer_size(res),
                 !!(etna_resource(res)->bo && getenv("ETNA_ML_CACHED")), mlt_now() - out_t0);
   }

   if (ETNA_ML_ENV("ETNA_ML_TIMING")) {
      long end = mlt_now();
      fprintf(stderr, "MLOUTS %u inputs-bytes-n/a\n", outputs_count);
      fprintf(stderr, "MLTIME k %d input %ld emit %ld gap %ld flush %ld wait %ld out %ld total %ld\n",
              subgraph->prefix_limit, mlt_t[1] - mlt_t[0], mlt_t[2] - mlt_t[1], mlt_t[3] - mlt_t[2], mlt_t[4] - mlt_t[3],
              mlt_t[5] - mlt_t[4], end - mlt_t[5], end - mlt_t[0]);
   }
}

void
etna_ml_subgraph_destroy(struct pipe_ml_device *pdevice, struct pipe_ml_subgraph *psubgraph)
{
   struct etna_ml_subgraph *subgraph = (struct etna_ml_subgraph *)(psubgraph);

   util_dynarray_foreach(&subgraph->operations, struct etna_vip_instruction, operation) {
      for (unsigned j = 0; j < MAX_CONFIG_BOS && operation->configs[j]; j++)
         etna_bo_del(operation->configs[j]);
      etna_bo_del(operation->coefficients);
      etna_bo_del(operation->pwl_lut);
      pipe_resource_reference(&operation->input, NULL);
      pipe_resource_reference(&operation->output, NULL);
   }
   util_dynarray_fini(&subgraph->operations);
   util_dynarray_fini(&subgraph->cpu_transposes);
   util_dynarray_fini(&subgraph->cpu_detransposes);
   util_dynarray_fini(&subgraph->cmd_bos);
   free(subgraph->cmd_cache);
   if (subgraph->sram_bo)
      etna_bo_del(subgraph->sram_bo);

   util_dynarray_foreach(&subgraph->tensors, struct etna_ml_tensor*, tensor) {
      if (!*tensor)
         continue;
      pipe_resource_reference(&(*tensor)->resource, NULL);
      free(*tensor);
   }
   util_dynarray_fini(&subgraph->tensors);

   free(subgraph);
}
