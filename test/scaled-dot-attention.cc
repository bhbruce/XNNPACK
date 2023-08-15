// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate.
#include <array>      // For std::array.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <limits>     // For std::numeric_limits.
#include <memory>     // For std::unique_ptr.
#include <numeric>    // For std::accumulate.
#include <random>     // For std::random_device, std::mt19937, std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/node-type.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

template <class T>
class ScaledDotAttentionTestBase : public ::testing::Test {
 protected:
  ScaledDotAttentionTestBase()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    dim_dist = std::uniform_int_distribution<size_t>(5, 15);
    bernoulli_dist = std::bernoulli_distribution(0.5);
    cap_dist = std::uniform_real_distribution<float>(1.0f, 50.0f);

    const size_t num_query_dims = 4;
    // Query is [N, H, T, C].
    query_dims = RandomShape(num_query_dims);
    batch_size = query_dims[0];
    query_heads = query_dims[1];
    query_tokens = query_dims[2];
    channels = query_dims[3];

    // Key/Value is [N, H, U, C].
    const size_t num_key_value_dims = 4;
    key_dims = query_dims;
    const bool test_multi_query = bernoulli_dist(rng);
    if (test_multi_query) {
      key_dims[1] = 1;
    }
    key_value_heads = key_dims[1];
    // Change key_value_tokens dim.
    const size_t key_value_tokens_dim = num_key_value_dims - 2;
    key_value_tokens = dim_dist(rng);
    key_dims[key_value_tokens_dim] = key_value_tokens;

    value_dims = key_dims;
    // Mask is [T, U].
    mask_dims = {query_tokens, key_value_tokens};
    // Mask is [C].
    scale_dims = {channels};
    // Output is [N, H, T, C].
    output_dims = query_dims;

    cap_type = bernoulli_dist(rng) ? xnn_attention_logits_cap_type_none : xnn_attention_logits_cap_type_tanh;
    if (cap_type == xnn_attention_logits_cap_type_tanh) {
      cap_params.cap = cap_dist(rng);
    } else {
      cap_params.cap = 0.0f;
    }

    query = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(query_dims));
    key = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(key_dims));
    value = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(value_dims));
    scale = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(scale_dims));
    mask = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(mask_dims));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(operator_output.size());
  };

  std::vector<size_t> RandomShape(size_t num_dims)
  {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  size_t NumElements(std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> cap_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::bernoulli_distribution bernoulli_dist;

  size_t batch_size;
  size_t query_heads;
  size_t query_tokens;
  size_t key_value_heads;
  size_t key_value_tokens;
  size_t channels;

  xnn_attention_logits_cap_type cap_type;
  xnn_attention_logits_cap_tanh_params cap_params;

  std::vector<size_t> query_dims;
  std::vector<size_t> key_dims;
  std::vector<size_t> value_dims;
  std::vector<size_t> scale_dims;
  std::vector<size_t> mask_dims;
  std::vector<size_t> output_dims;

  std::vector<T> query;
  std::vector<T> key;
  std::vector<T> value;
  std::vector<T> scale;
  std::vector<T> mask;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using ScaledDotAttentionTestF32 = ScaledDotAttentionTestBase<float>;

TEST_F(ScaledDotAttentionTestF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_attention(
      subgraph, cap_type, cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_scaled_dot_attention);
  EXPECT_EQ(node->compute_type, xnn_compute_type_fp32);
  EXPECT_EQ(node->num_inputs, 5);
  EXPECT_EQ(node->inputs[0], query_id);
  EXPECT_EQ(node->inputs[1], key_id);
  EXPECT_EQ(node->inputs[2], value_id);
  EXPECT_EQ(node->inputs[3], scale_id);
  EXPECT_EQ(node->inputs[4], mask_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->params.scaled_dot_attention.cap_type, cap_type);
  EXPECT_EQ(node->params.scaled_dot_attention.cap_tanh_params.cap, cap_params.cap);
  EXPECT_EQ(node->flags, 0);
}

TEST_F(ScaledDotAttentionTestF32, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;
  std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
  std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
  std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
  std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
  std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_scaled_dot_attention_nhtc_f32(cap_type, &cap_params, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_scaled_dot_attention_nhtc_f32(
                          op, batch_size, query_heads, query_tokens, key_value_heads, key_value_tokens, channels,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_scaled_dot_attention_nhtc_f32(op, workspace.data(), query.data(), key.data(), value.data(), scale.data(),
                                            mask.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_scaled_dot_attention(
      subgraph, cap_type, cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 6> external = {
    xnn_external_value{query_id, query.data()},
    xnn_external_value{key_id, key.data()},
    xnn_external_value{value_id, value.data()},
    xnn_external_value{scale_id, scale.data()},
    xnn_external_value{mask_id, mask.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    EXPECT_EQ(subgraph_output[i], operator_output[i]) << i;
  }
}

namespace {
void DefineScaledDotAttentionSubgraph(
  xnn_status* status_out,
  xnn_attention_logits_cap_type cap_type,
  xnn_attention_logits_cap_tanh_params cap_params,
  std::vector<size_t> query_dims,
  std::vector<size_t> key_dims,
  std::vector<size_t> value_dims,
  std::vector<size_t> scale_dims,
  std::vector<size_t> mask_dims,
  std::vector<size_t> output_dims,
  uint32_t batch_matrix_multiply_flags = 0)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(6, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t query_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, query_dims.size(), query_dims.data(), nullptr, /*external_id=*/0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_id));
  ASSERT_NE(query_id, XNN_INVALID_VALUE_ID);

  uint32_t key_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_dims.size(), key_dims.data(), nullptr, /*external_id=*/1,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_id));
  ASSERT_NE(key_id, XNN_INVALID_VALUE_ID);

  uint32_t value_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_dims.size(), value_dims.data(), nullptr, /*external_id=*/2,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_id));
  ASSERT_NE(value_id, XNN_INVALID_VALUE_ID);

  uint32_t scale_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, scale_dims.size(), scale_dims.data(), nullptr, /*external_id=*/3,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &scale_id));
  ASSERT_NE(scale_id, XNN_INVALID_VALUE_ID);

  uint32_t mask_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, mask_dims.size(), mask_dims.data(), nullptr, /*external_id=*/4,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &mask_id));
  ASSERT_NE(mask_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/5,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  *status_out = xnn_define_scaled_dot_attention(
    subgraph, cap_type, cap_params, query_id, key_id, value_id, scale_id, mask_id, output_id, /*flags=*/0);
}
}  // namespace

TEST(ScaledDotAttentionTest, cap_none_cap_value_is_zero) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params { 5.0f };
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, cap_tanh_cap_value_is_finite) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_tanh;
  xnn_attention_logits_cap_tanh_params cap_params { std::numeric_limits<float>::infinity() };
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, cap_tanh_cap_value_is_gt_zero) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_tanh;
  xnn_attention_logits_cap_tanh_params cap_params { 0.0f };
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, query_num_dim_ne_4_fails) {
  std::vector<size_t> query_dims = {2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, key_num_dim_ne_4_fails) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, value_num_dim_ne_4_fails) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, query_channels_eq_key_channels) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 7};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, query_heads_eq_key_heads) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 7, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, query_heads_eq_value_heads) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 7, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, query_channels_eq_value_channels) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 7};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, key_dims_not_equals_to_value_dims_fails) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, scale_num_dims_must_be_1) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5, 7};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, scale_channels_eq_query_channels) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {7};
  std::vector<size_t> mask_dims = {3, 3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, mask_num_dims_must_be_2) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 3, 5};
  std::vector<size_t> value_dims = {1, 2, 3, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, mask_query_tokens_eq_query_tokens) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {9, 7};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, mask_key_value_tokens_eq_key_value_tokens) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 9};
  std::vector<size_t> output_dims = {1, 2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}

TEST(ScaledDotAttentionTest, output_dims_eq_query_dims) {
  std::vector<size_t> query_dims = {1, 2, 3, 5};
  std::vector<size_t> key_dims = {1, 2, 7, 5};
  std::vector<size_t> value_dims = {1, 2, 7, 5};
  std::vector<size_t> scale_dims = {5};
  std::vector<size_t> mask_dims = {3, 7};
  std::vector<size_t> output_dims = {1, 2, 3, 7};
  xnn_status status = xnn_status_success;
  xnn_attention_logits_cap_type cap_type = xnn_attention_logits_cap_type_none;
  xnn_attention_logits_cap_tanh_params cap_params{};
  DefineScaledDotAttentionSubgraph(
    &status, cap_type, cap_params, query_dims, key_dims, value_dims, scale_dims, mask_dims, output_dims);
  EXPECT_EQ(xnn_status_invalid_parameter, status);
}
