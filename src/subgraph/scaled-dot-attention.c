// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_scaled_dot_attention_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  struct xnn_weights_cache* weights_cache)
{
  assert(node->num_inputs == 5);
  assert(node->num_outputs == 1);

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp32:
    {
      status = xnn_create_scaled_dot_attention_nhtc_f32(
        node->params.scaled_dot_attention.cap_type,
        &node->params.scaled_dot_attention.cap_tanh_params,
        /*flags=*/0,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_scaled_dot_attention_operator(
  struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t query_id = opdata->inputs[0];
  assert(query_id != XNN_INVALID_VALUE_ID);
  assert(query_id < num_values);
  const struct xnn_value* query = values + query_id;

  const uint32_t key_id = opdata->inputs[1];
  assert(key_id != XNN_INVALID_VALUE_ID);
  assert(key_id < num_values);
  const struct xnn_value* key = values + key_id;

  const uint32_t value_id = opdata->inputs[2];
  assert(value_id != XNN_INVALID_VALUE_ID);
  assert(value_id < num_values);
  const struct xnn_value* value = values + value_id;

  const uint32_t scale_id = opdata->inputs[3];
  assert(scale_id != XNN_INVALID_VALUE_ID);
  assert(scale_id < num_values);
  const struct xnn_value* scale = values + scale_id;

  const uint32_t mask_id = opdata->inputs[4];
  assert(mask_id != XNN_INVALID_VALUE_ID);
  assert(mask_id < num_values);
  const struct xnn_value* mask = values + mask_id;

  const size_t batch_size = query->shape.dim[0];
  const size_t query_heads = query->shape.dim[1];
  const size_t query_tokens = query->shape.dim[2];
  const size_t channels = query->shape.dim[3];
  const size_t key_heads = key->shape.dim[1];
  const size_t key_tokens = key->shape.dim[2];
  const size_t value_heads = key->shape.dim[1];
  const size_t value_tokens = key->shape.dim[2];

  if (key->shape.dim[0] != batch_size) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key batch size (%zu) must be equals to query batch size "
      "(%zu)", xnn_node_type_to_string(opdata->type), key_id, key->shape.dim[0], batch_size);
    return xnn_status_invalid_parameter;
  }

  if (key_heads != query_heads && key_heads != 1) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key heads (%zu) must be either equals to query heads or 1"
      " (%zu)", xnn_node_type_to_string(opdata->type), key_id, key_heads, query_heads);
    return xnn_status_invalid_parameter;
  }

  if (key->shape.dim[3] != channels) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key channels (%zu) must be equals to query channels"
      " (%zu)", xnn_node_type_to_string(opdata->type), key_id, key->shape.dim[3], channels);
    return xnn_status_invalid_parameter;
  }

  if (value->shape.dim[0] != batch_size) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value batch size (%zu) must be equals to query batch"
      "size (%zu)", xnn_node_type_to_string(opdata->type), value_id, value->shape.dim[0], batch_size);
    return xnn_status_invalid_parameter;
  }

  if (value_heads != query_heads && value_heads != 1) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value heads (%zu) must be either equals to query heads or 1"
      " (%zu)", xnn_node_type_to_string(opdata->type), value_id, value_heads, query_heads);
    return xnn_status_invalid_parameter;
  }

  if (value->shape.dim[3] != channels) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value channels (%zu) must be equals to query channels"
      " (%zu)", xnn_node_type_to_string(opdata->type), value_id, value->shape.dim[3], channels);
    return xnn_status_invalid_parameter;
  }

  if (key_heads != value_heads) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32" and value ID #%" PRIu32 ": key heads (%zu) must be equal "
      "to value heads (%zu)", xnn_node_type_to_string(opdata->type), key_id, value_id, key_heads, value_heads);
    return xnn_status_invalid_parameter;
  }

  if (key_tokens != value_tokens) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32" and value ID #%" PRIu32 ": key tokens (%zu) must be equal "
      "to value tokens (%zu)", xnn_node_type_to_string(opdata->type), key_id, value_id, key_tokens, value_tokens);
    return xnn_status_invalid_parameter;
  }

  if (scale->shape.dim[0] != channels) {
    xnn_log_error(
      "failed to define %s operator with scale ID #%" PRIu32 ": scale channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(opdata->type), scale_id, scale->shape.dim[0], channels);
    return xnn_status_invalid_parameter;
  }

  if (mask->shape.dim[0] != query_tokens) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask query tokens (%zu) must be equal to query tokens "
      "(%zu)", xnn_node_type_to_string(opdata->type), mask_id, mask->shape.dim[0], query_tokens);
    return xnn_status_invalid_parameter;
  }

  if (mask->shape.dim[1] != key_tokens) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask key/value tokens (%zu) must be equal to key/value "
      "tokens (%zu)", xnn_node_type_to_string(opdata->type), mask_id, mask->shape.dim[1], key_tokens);
    return xnn_status_invalid_parameter;
  }

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_scaled_dot_attention_nhtc_f32:
      return xnn_reshape_scaled_dot_attention_nhtc_f32(
        opdata->operator_objects[0],
        batch_size,
        query_heads,
        query_tokens,
        key_heads,
        key_tokens,
        channels,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_scaled_dot_attention_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t query_id = opdata->inputs[0];
  assert(query_id != XNN_INVALID_VALUE_ID);
  assert(query_id < num_values);
  const struct xnn_value* query = values + query_id;
  const void* query_data = query->data;
  assert(query_data != NULL);

  const uint32_t key_id = opdata->inputs[1];
  assert(key_id != XNN_INVALID_VALUE_ID);
  assert(key_id < num_values);
  const struct xnn_value* key = values + key_id;
  const void* key_data = key->data;
  assert(key_data != NULL);

  const uint32_t value_id = opdata->inputs[2];
  assert(value_id != XNN_INVALID_VALUE_ID);
  assert(value_id < num_values);
  const struct xnn_value* value = values + value_id;
  const void* attention_value_data = value->data;
  assert(attention_value_data != NULL);

  const uint32_t scale_id = opdata->inputs[3];
  assert(scale_id != XNN_INVALID_VALUE_ID);
  assert(scale_id < num_values);
  const struct xnn_value* scale = values + scale_id;
  const void* scale_data = scale->data;
  assert(scale_data != NULL);

  const uint32_t mask_id = opdata->inputs[4];
  assert(mask_id != XNN_INVALID_VALUE_ID);
  assert(mask_id < num_values);
  const struct xnn_value* mask = values + mask_id;
  const void* mask_data = mask->data;
  assert(mask_data != NULL);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  const struct xnn_value* output = values + output_id;
  void* output_data = output->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_scaled_dot_attention_nhtc_f32:
      return xnn_setup_scaled_dot_attention_nhtc_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        query_data,
        key_data,
        attention_value_data,
        scale_data,
        mask_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status check_inputs(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  const char* input_name)
{
  const enum xnn_node_type node_type = xnn_node_type_scaled_dot_attention;
  enum xnn_status status = xnn_subgraph_check_input_node_id(node_type, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with %s ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), input_name, input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  return status;
}

enum xnn_status xnn_define_scaled_dot_attention(
  xnn_subgraph_t subgraph,
  enum xnn_attention_logits_cap_type cap_type,
  struct xnn_attention_logits_cap_tanh_params cap_tanh_params,
  uint32_t query_id,
  uint32_t key_id,
  uint32_t value_id,
  uint32_t scale_id,
  uint32_t mask_id,
  uint32_t output_id,
  uint32_t flags)
{
  const enum xnn_node_type node_type = xnn_node_type_scaled_dot_attention;
  enum xnn_status status = xnn_subgraph_check_xnnpack_initialized(node_type);
  if (status != xnn_status_success) {
    return status;
  }

  if (cap_type == xnn_attention_logits_cap_type_none && cap_tanh_params.cap != 0.0f) {
    xnn_log_error(
        "failed to define %s operator with no logits cap: cap must be zero", xnn_node_type_to_string(node_type));
    return xnn_status_invalid_parameter;
  }

  const float cap_value = cap_tanh_params.cap;
  if (cap_type == xnn_attention_logits_cap_type_tanh && (!isfinite(cap_value) || cap_value <= 0.0f)) {
    xnn_log_error("failed to define %s operator with logits cap tanh: cap (%f) must be finite and positive",
                  xnn_node_type_to_string(node_type), cap_value);
    return xnn_status_invalid_parameter;
  }

  // Query is [N, H, T, C].
  status = check_inputs(subgraph, query_id, "query");
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* query = &subgraph->values[query_id];

  if (query->shape.num_dims != 4) {
    xnn_log_error(
      "failed to define %s operator with query ID #%" PRIu32 ": query must have 4 dimensions",
      xnn_node_type_to_string(node_type), query_id);
    return xnn_status_invalid_parameter;
  }

  const size_t batch_size = query->shape.dim[0];
  const size_t heads = query->shape.dim[1];
  const size_t query_tokens = query->shape.dim[2];
  const size_t channels = query->shape.dim[3];

  // Key can be [N, H, U, C] (multi-head) or [N, 1, U, C] (multi-query).
  status = check_inputs(subgraph, key_id, "key");
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* key = &subgraph->values[key_id];

  // Key must have 4 dimensions.
  if (key->shape.num_dims != 4) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key must have 4 dimensions",
      xnn_node_type_to_string(node_type), key_id);
    return xnn_status_invalid_parameter;
  }

  // Key batch size must match query.
  if (key->shape.dim[0] != batch_size) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key batch size (%zu) must be equal to query batch size "
      "(%zu)", xnn_node_type_to_string(node_type), key_id, key->shape.dim[0], batch_size);
    return xnn_status_invalid_parameter;
  }

  // Key heads must match query or be 1.
  if (key->shape.dim[1] != heads && key->shape.dim[1] != 1) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key heads (%zu) must be either equal to query heads "
      "(%zu) or 1", xnn_node_type_to_string(node_type), key_id, key->shape.dim[1], heads);
    return xnn_status_invalid_parameter;
  }

  // Key channels must match query.
  if (key->shape.dim[3] != channels) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(node_type), key_id, key->shape.dim[3], channels);
    return xnn_status_invalid_parameter;
  }

  // Value can be [N, H, U, C] (multi-head) or [N, 1, U, C] (multi-query).
  status = check_inputs(subgraph, value_id, "value");
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* value = &subgraph->values[value_id];

  // Value must have 4 dimensions.
  if (value->shape.num_dims != 4) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value must have 4 dimensions",
      xnn_node_type_to_string(node_type), value_id);
    return xnn_status_invalid_parameter;
  }

  // Value batch size must match query.
  if (value->shape.dim[0] != batch_size) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value batch size (%zu) must be either equal to query "
      "batch size (%zu)", xnn_node_type_to_string(node_type), value_id, value->shape.dim[0], batch_size);
    return xnn_status_invalid_parameter;
  }

  // Value heads must match query or be 1.
  if (value->shape.dim[1] != heads && value->shape.dim[1] != 1) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value heads (%zu) must be either equal to query heads "
      "(%zu) or 1", xnn_node_type_to_string(node_type), value_id, value->shape.dim[1], heads);
    return xnn_status_invalid_parameter;
  }

  // Value channels must match query.
  if (value->shape.dim[3] != channels) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(node_type), value_id, value->shape.dim[3], channels);
    return xnn_status_invalid_parameter;
  }

  // Check that key and value have the same dimensions.
  status = xnn_subgraph_check_all_dims_match(node_type, key_id, key, value_id, value);
  if (status != xnn_status_success) {
    return status;
  }

  // Scale is [C].
  status = check_inputs(subgraph, scale_id, "scale");
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* scale = &subgraph->values[scale_id];

  // Scale must have 1 dimension.
  if (scale->shape.num_dims != 1) {
    xnn_log_error(
      "failed to define %s operator with scale ID #%" PRIu32 ": scale must have only 1 dimension, found %zu",
        xnn_node_type_to_string(node_type), scale_id, scale->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Scale channels must match query channels.
  if (scale->shape.dim[0] != channels) {
    xnn_log_error(
      "failed to define %s operator with scale ID #%" PRIu32 ": scale channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(node_type), scale_id, scale->shape.dim[0], channels);
    return xnn_status_invalid_parameter;
  }

  // Mask is [T, U].
  status = check_inputs(subgraph, mask_id, "mask");
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* mask = &subgraph->values[mask_id];

  // Mask must have 2 dimensions.
  if (mask->shape.num_dims != 2) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask must have only 2 dimension, found %zu",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Mask query tokens must match query tokens.
  if (mask->shape.dim[0] != query_tokens) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask query tokens (%zu) must match query (%zu)",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.dim[0], query_tokens);
    return xnn_status_invalid_parameter;
  }

  // Mask key/value tokens must match key/value tokens.
  if (mask->shape.dim[1] != key->shape.dim[2]) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask key/value tokens (%zu) must match key/value (%zu)",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.dim[1], key->shape.dim[2]);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(node_type, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(node_type, output_id, output);
  if (status != xnn_status_success) {
    return status;
  }

  // Check that query and output have the same dimensions.
  status = xnn_subgraph_check_all_dims_match(node_type, query_id, query, output_id, output);
  if (status != xnn_status_success) {
    return status;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output->datatype) {
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), output_id,
        xnn_datatype_to_string(output->datatype), output->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = node_type;
  node->compute_type = compute_type;
  node->params.scaled_dot_attention.cap_type = cap_type;
  node->params.scaled_dot_attention.cap_tanh_params = cap_tanh_params;
  node->num_inputs = 5;
  node->inputs[0] = query_id;
  node->inputs[1] = key_id;
  node->inputs[2] = value_id;
  node->inputs[3] = scale_id;
  node->inputs[4] = mask_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_scaled_dot_attention_operator;
  node->reshape = reshape_scaled_dot_attention_operator;
  node->setup = setup_scaled_dot_attention_operator;

  return xnn_status_success;
}