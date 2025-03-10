// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>

#include <xnnpack/log.h>
#include <xnnpack/assembler.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/wasm-assembler.h>

namespace xnnpack {
namespace internal {

template <typename Generator>
xnn_status generate_gemm_or_igemm(xnn_code_buffer* b, const char* name, size_t max_mr, size_t kc,
                                  size_t loop_unroll_iters, bool full_unroll, size_t nc_mod_nr, const void* params) {
  size_t iters = kc / sizeof(float);
  assert(!full_unroll || (iters == loop_unroll_iters));
  Generator generator(b);

  generator.generate(name, max_mr, iters, loop_unroll_iters, full_unroll, nc_mod_nr,  static_cast<const jit_gemm_params*>(params));
  generator.Emit();
  auto finalized = generator.finalize();
  if (generator.error() == xnnpack::Error::kOutOfMemory) {
    return xnn_status_out_of_memory;
  }
  if (finalized == nullptr || generator.error() != xnnpack::Error::kNoError) {
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}

class PostOps : public WasmAssembler {
 public:
  using WasmAssembler::WasmAssembler;

  void InitPostOps(const jit_gemm_params* jit_gemm_params, Local& params) {
    InitClampLimit(jit_gemm_params->f32_minmax.min, jit_gemm_params->f32_minmax.max);
    InitNonClampPostOps(jit_gemm_params->post_operations, jit_gemm_params->num_post_operations, params);
  }

  void ApplyPostOps(LocalsArray& values) {
    Clamp(values);
    ApplyNonClampPostOps(values);
  }

 private:
  Local MakeV128Load64Splat(const Local& address, uint32_t offset) {
    return MakeLocal(V128Load64Splat(address, offset));
  }

  void InitClampLimit(float min, float max) {
    clamps_consts_.clamp_min = min != -std::numeric_limits<float>::infinity();
    clamps_consts_.clamp_max = max != +std::numeric_limits<float>::infinity();
    if (clamps_consts_.clamp_min) {
      clamps_consts_.vmin = MakeLocal(V128Const(min));
    }
    if (clamps_consts_.clamp_max) {
      clamps_consts_.vmax = MakeLocal(V128Const(max));
    }
  }

  void InitNonClampPostOps(const xnn_post_operation* ops, size_t num_ops, Local& params) {
    ops_ = ops;
    num_ops_ = num_ops;
    for (size_t i = 0; i < num_ops; i++) {
      switch (ops[i].op_type) {
        case xnn_post_operation_type_hardswish:
          hswish_consts_.vsixth = MakeV128Load64Splat(params, /*offset=*/0);
          hswish_consts_.vthree = MakeV128Load64Splat(params, /*offset=*/2 * sizeof(float));
          hswish_consts_.vsix = MakeV128Load64Splat(params, /*offset=*/4 * sizeof(float));
          hswish_consts_.vzero = MakeLocal(F32x4Splat(F32Const(0)));
          break;
        default:
          XNN_LOG_UNREACHABLE("unsupported post operation: %u", ops[i].op_type);
      }
      params = I32Add(params, I32Const(6 * sizeof(float)));
    }
  }

  void Clamp(Local& value) {
    if (clamps_consts_.clamp_max) {
      value = F32x4Pmin(clamps_consts_.vmax, value);
    }
    if (clamps_consts_.clamp_min) {
      value = F32x4Pmax(clamps_consts_.vmin, value);
    }
  }

  void Clamp(LocalsArray& values) {
    for (auto& value : values) Clamp(value);
  }

  void ApplyNonClampPostOps(Local& v) {
    for (size_t i = 0; i < num_ops_; i++) {
      switch (ops_[i].op_type) {
        case xnn_post_operation_type_hardswish:
          Hswish(v);
          break;
        default:
          XNN_LOG_UNREACHABLE("unsupported post operation: %u", ops[i].op_type);
      }
    }
  }

  void ApplyNonClampPostOps(LocalsArray& vs) {
    for (auto& v : vs) ApplyNonClampPostOps(v);
  }

  void Hswish(Local& v) {
    Local vacc = MakeLocal(F32x4Add(v, hswish_consts_.vthree));
    v = F32x4Mul(v, hswish_consts_.vsixth);
    vacc = F32x4Pmax(vacc, hswish_consts_.vzero);
    vacc = F32x4Pmin(vacc, hswish_consts_.vsix);
    v = F32x4Mul(vacc, v);
  }

  struct HswishConsts {
    Local vsixth;
    Local vsix;
    Local vthree;
    Local vzero;
  };

  struct ClampConsts {
    bool clamp_min{};
    bool clamp_max{};
    Local vmin;
    Local vmax;
  };

  const xnn_post_operation* ops_ = nullptr;
  size_t num_ops_{};
  HswishConsts hswish_consts_;
  ClampConsts clamps_consts_;
};


class GemmIGemmCommons : public PostOps {
 public:
  using PostOps::PostOps;

  void InitAccumulators(LocalsArray& vaccs, const Local& w, size_t offset) {
    vaccs[0] = V128Load(w, offset);
    std::for_each(std::next(std::begin(vaccs)), std::end(vaccs), [&](auto& vacc) { vacc = vaccs[0]; });
  }

  void ClampAsAndCs(LocalsArray& as, LocalsArray& cs, const Local& mr, const Local& a, const Local& c,
                    const Local& a_stride, const Local& cm_stride) {
    as[0] = a;
    cs[0] = c;
    auto i_local = MakeLocal(I32Const(1));
    for (size_t i = 1; i < as.size(); i++) {
      as[i] = I32Add(as[i - 1], a_stride);
      cs[i] = I32Add(cs[i - 1], cm_stride);
      If([&] { I32GeU(i_local, mr); },
         [&] {
           as[i] = as[i - 1];
           cs[i] = cs[i - 1];
         });
      i_local = I32Add(i_local, I32Const(1));
    }
  }

  void ClampCs(LocalsArray& cs, const Local& mr, const Local& c, const Local& cm_stride) {
    cs[0] = c;
    for (size_t i = 1; i < cs.size(); i++) {
      cs[i] = Select(cs[i - 1], I32Add(cs[i - 1], cm_stride), I32GeU(I32Const(i), mr));
    }
  }
};

}
}
