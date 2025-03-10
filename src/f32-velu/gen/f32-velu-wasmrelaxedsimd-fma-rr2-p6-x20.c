// Auto-generated file. Do not edit!
//   Template: src/f32-velu/wasmsimd-rr2-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>


void xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vprescale = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.prescale);
  const v128_t valpha = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.alpha);
  const v128_t vbeta = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.beta);
  const v128_t vsat_cutoff = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.sat_cutoff);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.magic_bias);
  const v128_t vlog2e = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.log2e);
  const v128_t vminus_ln2_hi = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.minus_ln2_hi);
  const v128_t vminus_ln2_lo = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.minus_ln2_lo);
  const v128_t vc6 = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.c6);
  const v128_t vc5 = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.c5);
  const v128_t vc4 = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.c4);
  const v128_t vc3 = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.c3);
  const v128_t vc2 = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.c2);
  const v128_t vone = wasm_v128_load64_splat(params->wasmsimd_rr2_p6.one);

  for (; batch >= 20 * sizeof(float); batch -= 20 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(input);
    v128_t vx4567 = wasm_v128_load(input + 4);
    v128_t vx89AB = wasm_v128_load(input + 8);
    v128_t vxCDEF = wasm_v128_load(input + 12);
    v128_t vxGHIJ = wasm_v128_load(input + 16);
    input += 20;

    const v128_t vz0123 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx0123, vprescale));
    const v128_t vz4567 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx4567, vprescale));
    const v128_t vz89AB = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx89AB, vprescale));
    const v128_t vzCDEF = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vxCDEF, vprescale));
    const v128_t vzGHIJ = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vxGHIJ, vprescale));

    v128_t vn0123 = wasm_f32x4_relaxed_madd(vz0123, vlog2e, vmagic_bias);
    v128_t vn4567 = wasm_f32x4_relaxed_madd(vz4567, vlog2e, vmagic_bias);
    v128_t vn89AB = wasm_f32x4_relaxed_madd(vz89AB, vlog2e, vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_relaxed_madd(vzCDEF, vlog2e, vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_relaxed_madd(vzGHIJ, vlog2e, vmagic_bias);

    v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_hi, vz0123);
    v128_t vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_hi, vz4567);
    v128_t vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_hi, vz89AB);
    v128_t vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_hi, vzCDEF);
    v128_t vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_hi, vzGHIJ);

    vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_lo, vt0123);
    vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_lo, vt4567);
    vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_lo, vt89AB);
    vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_lo, vtCDEF);
    vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_lo, vtGHIJ);

    v128_t vp0123 = wasm_f32x4_relaxed_madd(vc6, vt0123, vc5);
    v128_t vp4567 = wasm_f32x4_relaxed_madd(vc6, vt4567, vc5);
    v128_t vp89AB = wasm_f32x4_relaxed_madd(vc6, vt89AB, vc5);
    v128_t vpCDEF = wasm_f32x4_relaxed_madd(vc6, vtCDEF, vc5);
    v128_t vpGHIJ = wasm_f32x4_relaxed_madd(vc6, vtGHIJ, vc5);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc4);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc4);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc4);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc4);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc4);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc3);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc3);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc3);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc3);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc3);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc2);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc2);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc2);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc2);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc2);

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);
    vp89AB = wasm_f32x4_mul(vp89AB, vt89AB);
    vpCDEF = wasm_f32x4_mul(vpCDEF, vtCDEF);
    vpGHIJ = wasm_f32x4_mul(vpGHIJ, vtGHIJ);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vs89AB = wasm_f32x4_sub(vs89AB, vone);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vsCDEF = wasm_f32x4_sub(vsCDEF, vone);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vsGHIJ = wasm_f32x4_sub(vsGHIJ, vone);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vt0123);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vt4567);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vt89AB);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vtCDEF);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vtGHIJ);

    const v128_t ve0123 = wasm_f32x4_mul(wasm_f32x4_add(vp0123, vs0123), valpha);
    const v128_t ve4567 = wasm_f32x4_mul(wasm_f32x4_add(vp4567, vs4567), valpha);
    const v128_t ve89AB = wasm_f32x4_mul(wasm_f32x4_add(vp89AB, vs89AB), valpha);
    const v128_t veCDEF = wasm_f32x4_mul(wasm_f32x4_add(vpCDEF, vsCDEF), valpha);
    const v128_t veGHIJ = wasm_f32x4_mul(wasm_f32x4_add(vpGHIJ, vsGHIJ), valpha);

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);
    const v128_t vsignm89AB = wasm_i32x4_shr(vx89AB, 31);
    vx89AB = wasm_f32x4_mul(vx89AB, vbeta);
    const v128_t vsignmCDEF = wasm_i32x4_shr(vxCDEF, 31);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vbeta);
    const v128_t vsignmGHIJ = wasm_i32x4_shr(vxGHIJ, 31);
    vxGHIJ = wasm_f32x4_mul(vxGHIJ, vbeta);

    const v128_t vy0123 = wasm_i32x4_relaxed_laneselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_i32x4_relaxed_laneselect(ve4567, vx4567, vsignm4567);
    const v128_t vy89AB = wasm_i32x4_relaxed_laneselect(ve89AB, vx89AB, vsignm89AB);
    const v128_t vyCDEF = wasm_i32x4_relaxed_laneselect(veCDEF, vxCDEF, vsignmCDEF);
    const v128_t vyGHIJ = wasm_i32x4_relaxed_laneselect(veGHIJ, vxGHIJ, vsignmGHIJ);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    wasm_v128_store(output + 12, vyCDEF);
    wasm_v128_store(output + 16, vyGHIJ);
    output += 20;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx, vprescale));

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vlog2e, vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc6, vt, vc5);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    const v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_relaxed_max(wasm_f32x4_mul(vx, vprescale), vsat_cutoff);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vlog2e, vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc6, vt, vc5);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}
