// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/unaligned.h>


void xnn_f16_f32acc_rsum_ukernel__f16c_x8(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m256 vacc0 = _mm256_setzero_ps();
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc0 = _mm256_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m128i vmask = _mm_loadu_si128((const __m128i*) ((uintptr_t) &params->avx.mask_table[7] - batch));
    const __m128i vh = _mm_castps_si128(_mm_maskload_ps((const float*) i, vmask));
    const __m256 vt = _mm256_cvtph_ps(vh);
    vacc0 = _mm256_add_ps(vacc0, vt);
    i = (const void*) ((uintptr_t) i + batch);
    if (batch & (1 * sizeof(uint16_t))) {
      const __m128i vh = _mm_insert_epi16(_mm_setzero_si128(), (int) unaligned_load_u16(i - 1), 0);
      const __m256 vt = _mm256_zextps128_ps256(_mm_cvtph_ps(vh));
      vacc0 = _mm256_add_ps(vacc0, vt);
    }
  }
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc0), _mm256_extractf128_ps(vacc0, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->avx.scale));
  const __m128i vout = _mm_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
  unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout, 0));
}
