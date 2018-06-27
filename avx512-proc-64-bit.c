//=== AVX512 implementation - 64-bit code ==================================

#if defined(AVX512)

void
AVX512_mandelbrot(float Re_min, float Re_max,
                  float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i;

    __m256i *ptr = (__m256i *) data;

    _mm256_zeroall();

    // step on Re and Im axis
    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m512 vec_threshold = _mm512_set1_ps(threshold);

    // 2. Cim
    __m512 Cim = _mm512_set1_ps(Im_min);
    __m512 Cre, Xre, Xim, Xre2, Xim2, Xrm, Xtt, cmp;
    __m512i itercount;

    // 3. Re advance every x iteration
    __m512 vec_dRe = _mm512_set1_ps(16 * dRe);

    // 4. Im advance every y iteration
    __m512 vec_dIm = _mm512_set1_ps(dIm);

    // calculations
    for (y = 0; y < height; y++) {

        Xtt = _mm512_setr_ps(0 * dRe, 1 * dRe, 2 * dRe, 3 * dRe, 4 * dRe, 5 * dRe, 6 * dRe, 7 * dRe,
                             8 * dRe, 9 * dRe, 10 * dRe, 11 * dRe, 12 * dRe, 13 * dRe, 14 * dRe, 15 * dRe);
        Cre = _mm512_set1_ps(Re_min);
        Cre = _mm512_add_ps(Cre, Xtt);

        for (x = 0; x < width; x += 16) {

            Xre2 = _mm512_mul_ps(Cre, Cre);
            Xim2 = _mm512_mul_ps(Cim, Cim);
            Xrm = _mm512_mul_ps(Cre, Cim);
            itercount = _mm512_setzero_si512();

            for (i = 0; i < maxiters; i++) {
                cmp = _mm512_add_ps(Xre2, Xim2);
                Xre = _mm512_add_ps(Cre, _mm512_sub_ps(Xre2, Xim2));
                cmp = _mm512_cmple_ps(cmp, vec_threshold);
                Xim = _mm512_add_ps(Cim, _mm512_add_ps(Xrm, Xrm));
                // sqr_dist < threshold => 8 elements vector
                if (_mm512_test_all_zero((__m512i) cmp))
                    break;
                itercount = _mm512_sub_epi32(itercount, (__m512i) cmp);
                Xre2 = _mm512_mul_ps(Xre, Xre);
                Xim2 = _mm512_mul_ps(Xim, Xim);
                Xrm = _mm512_mul_ps(Xre, Xim);
            }

            *ptr++ = _mm512_cvtepi32_epi16(itercount);

            // advance Cre vector
            Cre = _mm512_add_ps(Cre, vec_dRe);
        }

        // advance Cim vector
        Cim = _mm512_add_ps(Cim, vec_dIm);
    }
}

#if defined(FMA)

//=== FMA implementation - 64-bit code ==================================

void
AVX512_FMA_mandelbrot(float Re_min, float Re_max, float Im_min, float Im_max, float threshold, int maxiters, int width,
                      int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i, j;

    __m256i *ptr = (__m256i *) data;
    int miniters = maxiters & ~7;

    // step on Re and Im axis
    _mm256_zeroall();

    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m512 vec_threshold = _mm512_set1_ps(threshold);

    // 2. Cim
    __m512 Cim = _mm512_set1_ps(Im_min);

    // 3. Re advance every x iteration
    __m512 vec_dRe = _mm512_set1_ps(16 * dRe);

    // 4. Im advance every y iteration
    __m512 vec_dIm = _mm512_set1_ps(dIm);

    __m512i itercount;
    __m512 Xre2, Xim2, cmp, Xrm, Xre_s, Xim_s, Xre, Xim, Xtt, Cre;

    // calculations
    for (y = 0; y < height; y++) {

        Xtt = _mm512_setr_ps(0 * dRe, 1 * dRe, 2 * dRe, 3 * dRe, 4 * dRe, 5 * dRe, 6 * dRe, 7 * dRe,
                             8 * dRe, 9 * dRe, 10 * dRe, 11 * dRe, 12 * dRe, 13 * dRe, 14 * dRe, 15 * dRe);
        Cre = _mm512_set1_ps(Re_min);
        Cre = _mm512_add_ps(Cre, Xtt);

        for (x = 0; x < width; x += 16) {

            Xre = Cre;
            Xim = Cim;

            i = 0;
            while (i < miniters) {

                Xre_s = Xre;
                Xim_s = Xim;

                for (j = 0; j < 8; j++) {

                    Xrm = _mm512_mul_ps(Xre, Xim);
                    Xtt = _mm512_fmsub_ps(Xim, Xim, Cre);
                    Xrm = _mm512_add_ps(Xrm, Xrm);
                    Xim = _mm512_add_ps(Cim, Xrm);
                    Xre = _mm512_fmsub_ps(Xre, Xre, Xtt);
                }       // for

                cmp = _mm512_mul_ps(Xre, Xre);
                cmp = _mm512_fmadd_ps(Xim, Xim, cmp);
                cmp = _mm512_cmple_ps(cmp, vec_threshold);
                if (_mm512_test_all_one((__m512i) cmp)) {
                    i += 8;
                    continue;
                }
                Xre = Xre_s;
                Xim = Xim_s;
                break;
            }
            itercount = _mm512_set1_epi32(i);

            if (i < maxiters) {
                Xre2 = _mm512_mul_ps(Xre, Xre);
                Xim2 = _mm512_mul_ps(Xim, Xim);
                Xrm = _mm512_mul_ps(Xre, Xim);

                while (i++ < maxiters) {
                    cmp = _mm512_add_ps(Xre2, Xim2);
                    Xre = _mm512_add_ps(Cre, _mm512_sub_ps(Xre2, Xim2));
                    cmp = _mm512_cmple_ps(cmp, vec_threshold);
                    Xim = _mm512_add_ps(Cim, _mm512_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm512_test_all_zero((__m512i) cmp))
                        break;
                    itercount = _mm512_sub_epi32(itercount, (__m512i) cmp);
                    Xre2 = _mm512_mul_ps(Xre, Xre);
                    Xim2 = _mm512_mul_ps(Xim, Xim);
                    Xrm = _mm512_mul_ps(Xre, Xim);
                }
            }

            *ptr++ = _mm512_cvtepi32_epi16(itercount);

            // advance Cre vector
            Cre = _mm512_add_ps(Cre, vec_dRe);
        }

        // advance Cim vector
        Cim = _mm512_add_ps(Cim, vec_dIm);
    }
}

void
AVX512_FMA_STITCH_mandelbrot(float Re_min, float Re_max,
                             float Im_min, float Im_max, float threshold, int maxiters, int width, int height, uint16_t * data)
{
    float dRe, dIm;
    int x, y, i, j;

    __m256i *ptr = (__m256i *) data;
    int miniters = maxiters & ~7;

    _mm256_zeroall();

    // step on Re and Im axis
    dRe = (Re_max - Re_min) / width;
    dIm = (Im_max - Im_min) / height;

    // prepare vectors
    // 1. threshold
    __m512 vec_threshold = _mm512_set1_ps(threshold);

    // 3. Re advance every x iteration
    __m512 vec_dRe = _mm512_set1_ps(16 * dRe);

    // 5. temp vectors
    __m512 Xre2, Xim2, Xrm;

    // calculations
#pragma omp parallel for
    for (y = 0; y < height; y += 2) {

        __m512 Cim0 = _mm512_add_ps(_mm512_set1_ps(Im_min), _mm512_set1_ps(y * dIm));
        __m512 Cim1 = _mm512_add_ps(Cim0, _mm512_set1_ps(dIm));
        __m512 Xtt = _mm512_setr_ps(0 * dRe, 1 * dRe, 2 * dRe, 3 * dRe, 4 * dRe, 5 * dRe, 6 * dRe, 7 * dRe,
                                    8 * dRe, 9 * dRe, 10 * dRe, 11 * dRe, 12 * dRe, 13 * dRe, 14 * dRe, 15 * dRe);

        __m256i *ptr0 = ptr + y * width / 32;
        __m256i *ptr1 = ptr0 + width / 32;

        __m512 Cre = _mm512_set1_ps(Re_min);

        Cre = _mm512_add_ps(Cre, Xtt);

        for (x = 0; x < width; x += 16) {

            __m512i itercount0, itercount1;
            __m512 cmp0, cmp1, Xrm0, Xrm1, Xtt0, Xtt1;
            __m512 Xre_s0, Xre_s1, Xim_s0, Xim_s1;
            __m512 Xre0 = Cre;
            __m512 Xim0 = Cim0;
            __m512 Xre1 = Cre;
            __m512 Xim1 = Cim1;

            i = 0;
            while (i < miniters) {

                Xre_s0 = Xre0;
                Xre_s1 = Xre1;
                Xim_s0 = Xim0;
                Xim_s1 = Xim1;

                for (j = 0; j < 8; j++) {

                    Xrm0 = _mm512_mul_ps(Xre0, Xim0);
                    Xrm1 = _mm512_mul_ps(Xre1, Xim1);
                    Xtt0 = _mm512_fmsub_ps(Xim0, Xim0, Cre);
                    Xtt1 = _mm512_fmsub_ps(Xim1, Xim1, Cre);
                    Xrm0 = _mm512_add_ps(Xrm0, Xrm0);
                    Xrm1 = _mm512_add_ps(Xrm1, Xrm1);
                    Xim0 = _mm512_add_ps(Cim0, Xrm0);
                    Xim1 = _mm512_add_ps(Cim1, Xrm1);
                    Xre0 = _mm512_fmsub_ps(Xre0, Xre0, Xtt0);
                    Xre1 = _mm512_fmsub_ps(Xre1, Xre1, Xtt1);
                }       // for

                cmp0 = _mm512_mul_ps(Xre0, Xre0);
                cmp1 = _mm512_mul_ps(Xre1, Xre1);
                cmp0 = _mm512_fmadd_ps(Xim0, Xim0, cmp0);
                cmp1 = _mm512_fmadd_ps(Xim1, Xim1, cmp1);
                cmp0 = _mm512_cmple_ps(cmp0, vec_threshold);
                cmp1 = _mm512_cmple_ps(cmp1, vec_threshold);
                if (_mm512_test_all_one(_mm512_and_si512((__m512i) cmp0, (__m512i) cmp1))) {
                    i += 8;
                    continue;
                }
                Xre0 = Xre_s0;
                Xre1 = Xre_s1;
                Xim0 = Xim_s0;
                Xim1 = Xim_s1;
                break;
            }
            itercount0 = _mm512_set1_epi32(i);
            itercount1 = itercount0;

            if (i < maxiters) {
                Xre2 = _mm512_mul_ps(Xre0, Xre0);
                Xim2 = _mm512_mul_ps(Xim0, Xim0);
                Xrm = _mm512_mul_ps(Xre0, Xim0);

                j = i;
                while (j++ < maxiters) {
                    cmp0 = _mm512_add_ps(Xre2, Xim2);
                    Xre0 = _mm512_add_ps(Cre, _mm512_sub_ps(Xre2, Xim2));
                    cmp0 = _mm512_cmple_ps(cmp0, vec_threshold);
                    Xim0 = _mm512_add_ps(Cim0, _mm512_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm512_test_all_zero((__m512i) cmp0))
                        break;
                    itercount0 = _mm512_sub_epi32(itercount0, (__m512i) cmp0);
                    Xre2 = _mm512_mul_ps(Xre0, Xre0);
                    Xim2 = _mm512_mul_ps(Xim0, Xim0);
                    Xrm = _mm512_mul_ps(Xre0, Xim0);
                }

                Xre2 = _mm512_mul_ps(Xre1, Xre1);
                Xim2 = _mm512_mul_ps(Xim1, Xim1);
                Xrm = _mm512_mul_ps(Xre1, Xim1);
                j = i;
                while (j++ < maxiters) {
                    cmp1 = _mm512_add_ps(Xre2, Xim2);
                    Xre1 = _mm512_add_ps(Cre, _mm512_sub_ps(Xre2, Xim2));
                    cmp1 = _mm512_cmple_ps(cmp1, vec_threshold);
                    Xim1 = _mm512_add_ps(Cim1, _mm512_add_ps(Xrm, Xrm));
                    // sqr_dist < threshold => 8 elements vector
                    if (_mm512_test_all_zero((__m512i) cmp1))
                        break;
                    itercount1 = _mm512_sub_epi32(itercount1, (__m512i) cmp1);
                    Xre2 = _mm512_mul_ps(Xre1, Xre1);
                    Xim2 = _mm512_mul_ps(Xim1, Xim1);
                    Xrm = _mm512_mul_ps(Xre1, Xim1);
                }

            }

            *ptr0++ = _mm512_cvtepi32_epi16(itercount0);
            *ptr1++ = _mm512_cvtepi32_epi16(itercount1);

            // advance Cre vector
            Cre = _mm512_add_ps(Cre, vec_dRe);
        }
    }
}

#endif
#endif
