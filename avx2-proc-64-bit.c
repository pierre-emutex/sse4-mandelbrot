//=== AVX2 implementation - 64-bit code ==================================
#include <immintrin.h>

void
AVX2_mandelbrot(float Re_min, float Re_max,
                float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i;

    uint64_t *ptr = (uint64_t *) data;

    _mm256_zeroall();

    // step on Re and Im axis
    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m256 vec_threshold = _mm256_set1_ps(threshold);

    // 2. Cim
    __m256 Cim = _mm256_set1_ps(Im_min);
    __m256 Cre, Xre, Xim, Xre2, Xim2, Xrm, cmp;
    __m256i itercount, itershuffle;

    // 3. Re advance every x iteration
    __m256 vec_dRe = _mm256_set1_ps(8 * dRe);

    // 4. Im advance every y iteration
    __m256 vec_dIm = _mm256_set1_ps(dIm);

    // calculations
    for (y = 0; y < height; y++) {

        Cre = _mm256_setr_ps(Re_min + 0 * dRe, Re_min + 1 * dRe, Re_min + 2 * dRe, Re_min + 3 * dRe,
                             Re_min + 4 * dRe, Re_min + 5 * dRe, Re_min + 6 * dRe, Re_min + 7 * dRe);

        for (x = 0; x < width; x += 8) {

            Xre2 = _mm256_mul_ps(Cre, Cre);
            Xim2 = _mm256_mul_ps(Cim, Cim);
            Xrm = _mm256_mul_ps(Cre, Cim);
            itercount = _mm256_setzero_si256();

            for (i = 0; i < maxiters; i++) {
                cmp = _mm256_add_ps(Xre2, Xim2);
                Xre = _mm256_add_ps(Cre, _mm256_sub_ps(Xre2, Xim2));
                cmp = _mm256_cmp_ps(cmp, vec_threshold, _CMP_LE_OS);
                Xim = _mm256_add_ps(Cim, _mm256_add_ps(Xrm, Xrm));
                // sqr_dist < threshold => 8 elements vector
                if (_mm256_testz_si256((__m256i) cmp, (__m256i) cmp))
                    break;
                itercount = _mm256_sub_epi32(itercount, (__m256i) cmp);
                Xre2 = _mm256_mul_ps(Xre, Xre);
                Xim2 = _mm256_mul_ps(Xim, Xim);
                Xrm = _mm256_mul_ps(Xre, Xim);
            }

            itershuffle = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           16, 17, 20, 21, 24, 25, 28, 29,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff);

            itercount = _mm256_shuffle_epi8(itercount, itershuffle);

            *ptr++ = _mm256_extract_epi64(itercount, 0);
            *ptr++ = _mm256_extract_epi64(itercount, 2);

            // advance Cre vector
            Cre = _mm256_add_ps(Cre, vec_dRe);
        }

        // advance Cim vector
        Cim = _mm256_add_ps(Cim, vec_dIm);
    }
}

#if defined(FMA)

//=== FMA implementation - 64-bit code ==================================

void
AVX2_FMA_mandelbrot(float Re_min, float Re_max,
                    float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i, j;

    uint64_t *ptr = (uint64_t *) data;
    int miniters = maxiters & ~7;

    // step on Re and Im axis
    _mm256_zeroall();

    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m256 vec_threshold = _mm256_set1_ps(threshold);
    __m256i vec_one = _mm256_set1_epi32(-1);

    // 2. Cim
    __m256 Cim = _mm256_set1_ps(Im_min);

    // 3. Re advance every x iteration
    __m256 vec_dRe = _mm256_set1_ps(8 * dRe);

    // 4. Im advance every y iteration
    __m256 vec_dIm = _mm256_set1_ps(dIm);

    __m256i itercount, itershuffle;
    __m256 Xre2, Xim2, cmp, Xrm, Xre_s, Xim_s, Xre, Xim, Xtt, Cre;

    // calculations
    for (y = 0; y < height; y++) {

        Xtt = _mm256_setr_ps(0 * dRe, 1 * dRe, 2 * dRe, 3 * dRe, 4 * dRe, 5 * dRe, 6 * dRe, 7 * dRe);
        Cre = _mm256_set1_ps(Re_min);
        Cre = _mm256_add_ps(Cre, Xtt);

        for (x = 0; x < width; x += 8) {

            Xre = Cre;
            Xim = Cim;

            i = 0;
            while (i < miniters) {

                Xre_s = Xre;
                Xim_s = Xim;

                for (j = 0; j < 8; j++) {

                    Xrm = _mm256_mul_ps(Xre, Xim);
                    Xtt = _mm256_fmsub_ps(Xim, Xim, Cre);
                    Xrm = _mm256_add_ps(Xrm, Xrm);
                    Xim = _mm256_add_ps(Cim, Xrm);
                    Xre = _mm256_fmsub_ps(Xre, Xre, Xtt);
                }       // for

                cmp = _mm256_mul_ps(Xre, Xre);
                cmp = _mm256_fmadd_ps(Xim, Xim, cmp);
                cmp = _mm256_cmp_ps(cmp, vec_threshold, _CMP_LE_OS);
                if (_mm256_testc_si256((__m256i) cmp, vec_one)) {
                    i += 8;
                    continue;
                }
                Xre = Xre_s;
                Xim = Xim_s;
                break;
            }
            itercount = _mm256_set1_epi32(i);

            if (i < maxiters) {
                Xre2 = _mm256_mul_ps(Xre, Xre);
                Xim2 = _mm256_mul_ps(Xim, Xim);
                Xrm = _mm256_mul_ps(Xre, Xim);

                while (i++ < maxiters) {
                    cmp = _mm256_add_ps(Xre2, Xim2);
                    Xre = _mm256_add_ps(Cre, _mm256_sub_ps(Xre2, Xim2));
                    cmp = _mm256_cmp_ps(cmp, vec_threshold, _CMP_LE_OS);
                    Xim = _mm256_add_ps(Cim, _mm256_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm256_testz_si256((__m256i) cmp, (__m256i) cmp))
                        break;
                    itercount = _mm256_sub_epi32(itercount, (__m256i) cmp);
                    Xre2 = _mm256_mul_ps(Xre, Xre);
                    Xim2 = _mm256_mul_ps(Xim, Xim);
                    Xrm = _mm256_mul_ps(Xre, Xim);
                }
            }

            itershuffle = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           16, 17, 20, 21, 24, 25, 28, 29,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff);

            itercount = _mm256_shuffle_epi8(itercount, itershuffle);

            *ptr++ = _mm256_extract_epi64(itercount, 0);
            *ptr++ = _mm256_extract_epi64(itercount, 2);

            // advance Cre vector
            Cre = _mm256_add_ps(Cre, vec_dRe);
        }

        // advance Cim vector
        Cim = _mm256_add_ps(Cim, vec_dIm);
    }
}

void
AVX2_FMA_STITCH_mandelbrot(float Re_min, float Re_max,
                           float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int y;

    int miniters = maxiters & ~7;

    _mm256_zeroall();

    // step on Re and Im axis
    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m256 vec_threshold = _mm256_set1_ps(threshold);
    __m256i vec_one = _mm256_set1_epi32(-1);

    // 3. Re advance every x iteration
    __m256 vec_dRe = _mm256_set1_ps(8 * dRe);

    // 5. temp vectors
    __m256 Xre2, Xim2, Xrm;

    // calculations
#pragma omp parallel for
    for (y = 0; y < height; y += 2) {

        __m256 Cim0 = _mm256_add_ps(_mm256_set1_ps(Im_min), _mm256_set1_ps(y * dIm));
        __m256 Cim1 = _mm256_add_ps(Cim0, _mm256_set1_ps(dIm));

        __m256 Xtt = _mm256_setr_ps(0 * dRe, 1 * dRe, 2 * dRe, 3 * dRe,
                                    4 * dRe, 5 * dRe, 6 * dRe, 7 * dRe);
	int x, i, j;
        uint64_t *ptr0 = (uint64_t *) (data + y * width);
        uint64_t *ptr1 = (uint64_t *) (data + y * width + width);

        __m256 Cre = _mm256_set1_ps(Re_min);

        Cre = _mm256_add_ps(Cre, Xtt);

        for (x = 0; x < width; x += 8) {

            __m256i itercount0, itercount1;
            __m256 cmp0, cmp1, Xrm0, Xrm1, Xtt0, Xtt1;
            __m256 Xre_s0, Xre_s1, Xim_s0, Xim_s1;
            __m256 Xre0 = Cre;
            __m256 Xim0 = Cim0;
            __m256 Xre1 = Cre;
            __m256 Xim1 = Cim1;

            i = 0;
            while (i < miniters) {

                Xre_s0 = Xre0;
                Xre_s1 = Xre1;
                Xim_s0 = Xim0;
                Xim_s1 = Xim1;

                for (j = 0; j < 8; j++) {

                    Xrm0 = _mm256_mul_ps(Xre0, Xim0);
                    Xrm1 = _mm256_mul_ps(Xre1, Xim1);
                    Xtt0 = _mm256_fmsub_ps(Xim0, Xim0, Cre);
                    Xtt1 = _mm256_fmsub_ps(Xim1, Xim1, Cre);
                    Xrm0 = _mm256_add_ps(Xrm0, Xrm0);
                    Xrm1 = _mm256_add_ps(Xrm1, Xrm1);
                    Xim0 = _mm256_add_ps(Cim0, Xrm0);
                    Xim1 = _mm256_add_ps(Cim1, Xrm1);
                    Xre0 = _mm256_fmsub_ps(Xre0, Xre0, Xtt0);
                    Xre1 = _mm256_fmsub_ps(Xre1, Xre1, Xtt1);
                }       // for

                cmp0 = _mm256_mul_ps(Xre0, Xre0);
                cmp1 = _mm256_mul_ps(Xre1, Xre1);
                cmp0 = _mm256_fmadd_ps(Xim0, Xim0, cmp0);
                cmp1 = _mm256_fmadd_ps(Xim1, Xim1, cmp1);
                cmp0 = _mm256_cmp_ps(cmp0, vec_threshold, _CMP_LE_OS);
                cmp1 = _mm256_cmp_ps(cmp1, vec_threshold, _CMP_LE_OS);
                if (_mm256_testc_si256((__m256i) _mm256_and_ps(cmp0, cmp1), vec_one)) {
                    i += 8;
                    continue;
                }
                Xre0 = Xre_s0;
                Xre1 = Xre_s1;
                Xim0 = Xim_s0;
                Xim1 = Xim_s1;
                break;
            }
            itercount0 = _mm256_set1_epi32(i);
            itercount1 = itercount0;

            if (i < maxiters) {
                Xre2 = _mm256_mul_ps(Xre0, Xre0);
                Xim2 = _mm256_mul_ps(Xim0, Xim0);
                Xrm = _mm256_mul_ps(Xre0, Xim0);

                j = i;
                while (j++ < maxiters) {
                    cmp0 = _mm256_add_ps(Xre2, Xim2);
                    Xre0 = _mm256_add_ps(Cre, _mm256_sub_ps(Xre2, Xim2));
                    cmp0 = _mm256_cmp_ps(cmp0, vec_threshold, _CMP_LE_OS);
                    Xim0 = _mm256_add_ps(Cim0, _mm256_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm256_testz_si256((__m256i) cmp0, (__m256i) cmp0))
                        break;
                    itercount0 = _mm256_sub_epi32(itercount0, (__m256i) cmp0);
                    Xre2 = _mm256_mul_ps(Xre0, Xre0);
                    Xim2 = _mm256_mul_ps(Xim0, Xim0);
                    Xrm = _mm256_mul_ps(Xre0, Xim0);
                }

                Xre2 = _mm256_mul_ps(Xre1, Xre1);
                Xim2 = _mm256_mul_ps(Xim1, Xim1);
                Xrm = _mm256_mul_ps(Xre1, Xim1);
                j = i;
                while (j++ < maxiters) {
                    cmp1 = _mm256_add_ps(Xre2, Xim2);
                    Xre1 = _mm256_add_ps(Cre, _mm256_sub_ps(Xre2, Xim2));
                    cmp1 = _mm256_cmp_ps(cmp1, vec_threshold, _CMP_LE_OS);
                    Xim1 = _mm256_add_ps(Cim1, _mm256_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm256_testz_si256((__m256i) cmp1, (__m256i) cmp1))
                        break;
                    itercount1 = _mm256_sub_epi32(itercount1, (__m256i) cmp1);
                    Xre2 = _mm256_mul_ps(Xre1, Xre1);
                    Xim2 = _mm256_mul_ps(Xim1, Xim1);
                    Xrm = _mm256_mul_ps(Xre1, Xim1);
                }

            }

            __m256i itershuffle = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           16, 17, 20, 21, 24, 25, 28, 29,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff,
                                           (char) 0xff, (char) 0xff, (char) 0xff, (char) 0xff);

            itercount0 = _mm256_shuffle_epi8(itercount0, itershuffle);
            itercount1 = _mm256_shuffle_epi8(itercount1, itershuffle);

            *ptr0++ = _mm256_extract_epi64(itercount0, 0);
            *ptr0++ = _mm256_extract_epi64(itercount0, 2);
            *ptr1++ = _mm256_extract_epi64(itercount1, 0);
            *ptr1++ = _mm256_extract_epi64(itercount1, 2);

            // advance Cre vector
            Cre = _mm256_add_ps(Cre, vec_dRe);
        }
    }
}

#endif
