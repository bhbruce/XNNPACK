// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packw.yaml
//   Generator: tools/generate-packw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(2)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(2)
      .k(10)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, n_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, n_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(4)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, n_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 1; n < 2; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, n_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 4; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 4; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, k_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(2)
      .k(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, k_div_2) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(2)
      .k(10)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, k_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 2; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, k_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 3; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, n_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, n_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(4)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, n_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 1; n < 2; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, n_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 4; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_X2_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 4; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_x2_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_X8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, n_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, n_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_X8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(4)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(20)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, n_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, n_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(24)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, n_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 12; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, n_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 13; n < 24; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(4)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(20)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, n_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, n_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(24)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, n_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 12; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, n_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 13; n < 24; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(8)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(40)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, n_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, n_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(24)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, n_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 12; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, n_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 13; n < 24; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(8)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(12)
      .k(40)
      .nr(12)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, n_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(12)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, n_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(24)
        .k(k)
        .nr(12)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, n_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 12; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, n_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 13; n < 24; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(12)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_X8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 13; n < 24; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(12)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, k_div_4) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X4_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, n_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, n_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_X8_PRFM, null_bias) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(2)
      .k(4)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(2)
      .k(20)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, n_eq_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, n_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(4)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, n_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 2; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, n_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(2)
      .k(4)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(2)
      .k(20)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, n_eq_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, n_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(4)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, n_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 2; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, n_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__SSE2_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__SSE2_X8_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__sse2_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(8)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(8)
      .k(40)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, n_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__SSE2_X8_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__SSE2_X8_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__sse2_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, k_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(8)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, k_div_8) {
    TEST_REQUIRES_X86_SSE2;
    PackWMicrokernelTester()
      .n(16)
      .k(40)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, k_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, k_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 9; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, n_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t k = 1; k < 16; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__SSE2_X8_PRFM, null_bias) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 16; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_x8_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, n_eq_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, n_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, n_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, n_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__AVX_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__avx_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, n_eq_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, n_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, n_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, n_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, n_eq_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, n_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, n_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, n_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__AVX_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, n_eq_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, n_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, n_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, n_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, n_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, n_eq_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, n_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, n_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, n_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_AVX;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, n_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16S4__AVX_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, k_eq_4) {
    TEST_REQUIRES_X86_AVX512F;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, k_div_4) {
    TEST_REQUIRES_X86_AVX512F;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, k_lt_4) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, k_gt_4) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, n_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, n_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, n_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, n_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, g_gt_1) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4, null_bias) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, k_eq_4) {
    TEST_REQUIRES_X86_AVX512F;
    PackWMicrokernelTester()
      .n(16)
      .k(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, k_div_4) {
    TEST_REQUIRES_X86_AVX512F;
    PackWMicrokernelTester()
      .n(16)
      .k(20)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, k_lt_4) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, k_gt_4) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, n_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, n_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(32)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, n_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, n_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, g_gt_1) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X16__AVX512F_X4_PRFM, null_bias) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 17; n < 32; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(16)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_x4_prfm);
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, k_eq_4) {
    PackWMicrokernelTester()
      .n(2)
      .k(4)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, k_div_4) {
    PackWMicrokernelTester()
      .n(2)
      .k(20)
      .nr(2)
      .kr(4)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, n_eq_2) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(2)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, n_div_2) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(4)
        .k(k)
        .nr(2)
        .kr(4)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, n_lt_2) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 2; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, n_gt_2) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(2)
          .kr(4)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, g_gt_1) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_X4, null_bias) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 3; n < 4; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(2)
            .kr(4)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, k_eq_4) {
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, k_div_4) {
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, n_eq_8) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, n_div_8) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, n_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, n_gt_8) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, g_gt_1) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8__WASMSIMD_X4, null_bias) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(1)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, k_eq_4) {
    PackWMicrokernelTester()
      .n(8)
      .k(4)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, k_div_4) {
    PackWMicrokernelTester()
      .n(8)
      .k(20)
      .nr(8)
      .kr(1)
      .sr(4)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, k_lt_4) {
    for (size_t k = 1; k < 4; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, k_gt_4) {
    for (size_t k = 5; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, n_eq_8) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(8)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, n_div_8) {
    for (size_t k = 1; k < 8; k++) {
      PackWMicrokernelTester()
        .n(16)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(4)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, n_lt_8) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 1; n < 8; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, n_gt_8) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(4)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, g_gt_1) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
        }
      }
    }
  }

  TEST(X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_X4, null_bias) {
    for (size_t g = 2; g <= 3; g++) {
      for (size_t k = 1; k < 8; k++) {
        for (size_t n = 9; n < 16; n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(2)
            .n(n)
            .k(k)
            .nr(8)
            .kr(1)
            .sr(4)
            .Test(xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_x4);
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(20)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, n_eq_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, n_div_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, n_lt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, n_gt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(2)
    .k(20)
    .nr(2)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, n_eq_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(2)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, n_div_2) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(2)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, n_lt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, n_gt_2) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(2)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X2__SCALAR_INT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 3; n < 4; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(2)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(3)
    .k(4)
    .nr(3)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(3)
    .k(20)
    .nr(3)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, n_eq_3) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, n_div_3) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(6)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, n_lt_3) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 3; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(3)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, n_gt_3) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 4; n < 6; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(3)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 4; n < 6; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(3)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 4; n < 6; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(3)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(3)
    .k(4)
    .nr(3)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(3)
    .k(20)
    .nr(3)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, n_eq_3) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(3)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, n_div_3) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(6)
      .k(k)
      .nr(3)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, n_lt_3) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 3; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(3)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, n_gt_3) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 4; n < 6; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(3)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 4; n < 6; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(3)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X3__SCALAR_INT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 4; n < 6; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(3)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(20)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, n_eq_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, n_div_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, n_lt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, n_gt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(4)
    .k(20)
    .nr(4)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_eq_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(4)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_div_4) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(4)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_lt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, n_gt_4) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(4)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X4__SCALAR_INT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 5; n < 8; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(4)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(20)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, n_eq_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, n_div_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, n_lt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, n_gt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(8)
    .k(20)
    .nr(8)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_eq_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(8)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_div_8) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(8)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_lt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 8; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, n_gt_8) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 9; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(8)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X8__SCALAR_INT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 9; n < 16; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(8)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(4)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(20)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, n_eq_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, n_div_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, n_lt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, n_gt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_x4);
      }
    }
  }
}


TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_eq_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(4)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_div_4) {
  PackWMicrokernelTester()
    .n(16)
    .k(20)
    .nr(16)
    .kr(1)
    .sr(1)
    .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, k_gt_4) {
  for (size_t k = 5; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_eq_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(16)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_div_16) {
  for (size_t k = 1; k < 8; k++) {
    PackWMicrokernelTester()
      .n(32)
      .k(k)
      .nr(16)
      .kr(1)
      .sr(1)
      .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_lt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 1; n < 16; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, n_gt_16) {
  for (size_t k = 1; k < 8; k++) {
    for (size_t n = 17; n < 32; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(16)
        .kr(1)
        .sr(1)
        .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, g_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .g(2)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
      }
    }
  }
}

TEST(X32_PACKW_GEMM_GOI_X16__SCALAR_INT_X4, null_bias) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < 8; k++) {
      for (size_t n = 17; n < 32; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(2)
          .n(n)
          .k(k)
          .nr(16)
          .kr(1)
          .sr(1)
          .Test(xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_x4);
      }
    }
  }
}
