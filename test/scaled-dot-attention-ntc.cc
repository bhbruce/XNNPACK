// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <gtest/gtest.h>

#include "attention-operator-tester.h"


TEST(ATTENTION_NHTC_F16, unit_batch) {
  AttentionOperatorTester()
      .batch_size(2)
      .query_tokens(41)
      .channels(137)
      .TestF16();
}

TEST(ATTENTION_NHTC_F16, batch_size) {
  AttentionOperatorTester()
      .batch_size(13)
      .query_tokens(41)
      .channels(137)
      .TestF16();
}

TEST(ATTENTION_NHTC_F16, large_channels) {
  AttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .channels(1543)
      .TestF16();
}

TEST(ATTENTION_NHTC_F16, self_attention_with_cap) {
  AttentionOperatorTester()
      .cap_tanh(20.0f)
      .query_tokens(41)
      .channels(137)
      .TestF16();
}

TEST(ATTENTION_NHTC_F16, cross_attention_key_value_tokens_lt_query_tokens) {
  AttentionOperatorTester()
      .query_tokens(41)
      .key_value_tokens(29)
      .channels(137)
      .TestF16();
}

TEST(ATTENTION_NHTC_F16, cross_attention_key_value_tokens_gt_query_tokens) {
  AttentionOperatorTester()
      .query_tokens(29)
      .key_value_tokens(41)
      .channels(137)
      .TestF16();
}

TEST(ATTENTION_NHTC_F32, unit_batch) {
  AttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, batch_size) {
  AttentionOperatorTester()
      .batch_size(13)
      .query_tokens(41)
      .channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, large_channels) {
  AttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .channels(1543)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, self_attention_with_cap) {
  AttentionOperatorTester()
      .cap_tanh(20.0f)
      .query_tokens(41)
      .channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, cross_attention_key_value_tokens_lt_query_tokens) {
  AttentionOperatorTester()
      .query_tokens(41)
      .key_value_tokens(29)
      .channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, cross_attention_key_value_tokens_gt_query_tokens) {
  AttentionOperatorTester()
      .query_tokens(29)
      .key_value_tokens(41)
      .channels(137)
      .TestF32();
}