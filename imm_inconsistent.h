/* 
*
* this file implements consistent naming for multiple SIMD variants (128, 256, 512 bits)
*
*/

#ifndef IMM_INCONSISTENT_H_H_
#define IMM_INCONSISTENT_H_H_

#include <immintrin.h>

// static inline __m128 _mm_cmple_ps(__m128 a, __m128 b)
// static inline __m128 _mm_cmpgt_ps(__m128 a, __m128 b)

#ifdef AVX2

static inline __m256 _mm256_cmple_ps(__m256 a, __m256 b)
{
       return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

static inline __m256 _mm256_cmpgt_ps(__m256 a, __m256 b)
{
       return _mm256_cmp_ps(a, b, _CMP_GT_OS);
}

#endif

#ifdef AVX512

#ifdef AVX512DQ

static inline __m512 _mm512_cmpgt_ps(__m512 a, __m512 b)
{
    return (__m512)_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GT_OS));
}

static inline __m512 _mm512_cmple_ps(__m512 a, __m512 b)
{
    return (__m512)_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LE_OS));
}

#else

static inline __m512 _mm512_cmpgt_ps(__m512 a, __m512 b)
{
    return (__m512)_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GT_OS), -1);
}

static inline __m512 _mm512_cmple_ps(__m512 a, __m512 b)
{
    return (__m512)_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LE_OS), -1);
}

#endif

#endif

#ifdef SSE4

int _mm_test_all_one (__m128i a)
{
	return _mm_testc_si128(a,_mm_set1_epi32(-1));
}

int _mm_test_all_zero (__m128i a)
{
	return _mm_testz_si128(a,a);
}

#endif

#ifdef AVX2

unsigned _mm256_test_all_one (__m256i a)
{
	return _mm256_testc_si256(a,_mm256_set1_epi32(-1));
}

unsigned _mm256_test_all_zero (__m256i a)
{
	return _mm256_testz_si256(a,a);
}

#endif

#ifdef AVX512

unsigned _mm512_test_all_one (__m512i a)
{
    return _mm512_cmpeq_epu64_mask(a, _mm512_set1_epi32(-1)) == 0xff;
}

unsigned _mm512_test_all_zero (__m512i a)
{
    return _mm512_testn_epi64_mask(a, a) == 0xff;
}

#endif

#endif

