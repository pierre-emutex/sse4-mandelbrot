//=== SSE4 implementation - 64-bit code ==================================
#include <immintrin.h>

void
SSE_mandelbrot(float Re_min, float Re_max,
               float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i;

    uint64_t *ptr = (uint64_t *) data;

    // step on Re and Im axis
    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m128 vec_threshold = _mm_set1_ps(threshold);
    __m128 Cre, Xre, Xim, Xre2, Xim2, Xrm, cmp;
    __m128i itercount;

    // 2. Cim
    __m128 Cim = _mm_set1_ps(Im_min);

    // 3. Re advance every x iteration
    __m128 vec_dRe = _mm_set1_ps(4 * dRe);

    // 4. Im advance every y iteration
    __m128 vec_dIm = _mm_set1_ps(dIm);

    // calculations
    for (y = 0; y < height; y++) {

        Cre = _mm_setr_ps(Re_min, Re_min + dRe, Re_min + 2 * dRe, Re_min + 3 * dRe);

        for (x = 0; x < width; x += 4) {

            Xre2 = _mm_mul_ps(Cre, Cre);
            Xim2 = _mm_mul_ps(Cim, Cim);
            Xrm = _mm_mul_ps(Cre, Cim);
            itercount = _mm_setzero_si128();

            for (i = 0; i < maxiters; i++) {
                cmp = _mm_add_ps(Xre2, Xim2);
                Xre = _mm_add_ps(Cre, _mm_sub_ps(Xre2, Xim2));
                cmp = _mm_cmple_ps(cmp, vec_threshold);
                Xim = _mm_add_ps(Cim, _mm_add_ps(Xrm, Xrm));
                // sqr_dist < threshold => 8 elements vector
                if (_mm_test_all_zero((__m128i) cmp))
                    break;
                itercount = _mm_sub_epi32(itercount, (__m128i) cmp);
                Xre2 = _mm_mul_ps(Xre, Xre);
                Xim2 = _mm_mul_ps(Xim, Xim);
                Xrm = _mm_mul_ps(Xre, Xim);
            }

            __m128i t1 = _mm_packus_epi32(itercount, itercount);
            *ptr++ = _mm_cvtsi128_si64(t1);

            // advance Cre vector
            Cre = _mm_add_ps(Cre, vec_dRe);
        }

        // advance Cim vector
        Cim = _mm_add_ps(Cim, vec_dIm);
    }
}
