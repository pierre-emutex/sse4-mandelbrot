
/*
	Mandelbrot fractal generator --- SSE2 & SSE4.1 implementation, $Revision: 1.3 $

	Author: Wojciech Muła
	e-mail: wojciech_mula@poczta.onet.pl
	www:    http://0x80.pl/

	License: BSD

	initial release 28-06-2008, last update $Date: 2008-12-22 20:15:08 $

	----------------------------------------------------------------------

	SSE procedure calculates 4 pixels in parallel. SSE4.1 procedure uses
	PTEST instruction to break loop when lengths of all 4 complex numbers
	are greater than some threshold.  SSE2 version uses PMOVMSKB and x86
	TEST.

	Average speedup over FPU procedure is around 4.5 times.
	Measured on Core2 Duo E8200 @ 2.66GHz.

	Usage:

		run program without arguments to read help

	----------------------------------------------------------------------

        Heavily modified by Pierre, 20-05-2018 

	Average speedup over original procedure is > 200 times.
		SSE4 computes 4 pixels in parallel
		AVX2 computes 8 pixels in parallel
		AVX512 computes 16 pixels in parallel
		stitched code computes 2 strands of 16 pixels in parallel
		execution on hyperthreads computes 4 strands of 16 pixels in parallel on the same core
		openmp scales on multiple cores

        Command line equivalent to original project

		 ./fractal64 -p ORIG -w 512 -h 512 -xmin -1 -xmax -1 -ymin 1 -ymax 1 -t 4.0 -i 255 -pgm

*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <stdarg.h>

//=== helper functions ===================================================
uint32_t
get_time(void)
{
    static struct timeval T;

    gettimeofday(&T, NULL);
    return (T.tv_sec * 1000000) + T.tv_usec;
}

void
die(const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vprintf(fmt, ap);
    putchar('\n');
    va_end(ap);

    exit(EXIT_FAILURE);
}

#include <immintrin.h>

#include "imm_inconsistent.h"

#include "fpu-proc.c"

#if defined(SSE4)
#include "sse4-proc-64-bit.c"
#endif

#if defined(AVX2)
#include "avx2-proc-64-bit.c"
#endif

#if defined(AVX512)
#include "avx512-proc-64-bit.c"
#endif
//=== Colors =============================================================


static unsigned all_colors[1024] = {
    0xffffffff, 0x80ee8011, 0x80ee8011, 0x81ed8110, 0x82ed8210, 0x83ec830f, 0x83ec830f, 0x84ec840f, 0x85eb850e, 0x86eb860e,
    0x87ea870e, 0x87ea870d, 0x88ea880d, 0x89e9890d, 0x8ae98a0c, 0x8ae88a0c, 0x8be88b0c, 0x8ce78c0b, 0x8de78d0b, 0x8ee68e0b,
    0x8ee68e0a, 0x8fe58f0a, 0x90e5900a, 0x91e49109, 0x91e49109, 0x92e49209, 0x93e39309, 0x94e39408, 0x95e29508, 0x95e29508,
    0x96e19607, 0x97e19707, 0x98e09807, 0x98e09807, 0x99df9906, 0x9ade9a06, 0x9bde9b06, 0x9bdd9b06, 0x9cdd9c05, 0x9ddc9d05,
    0x9edc9e05, 0x9edb9e05, 0x9fdb9f05, 0xa0daa004, 0xa1daa104, 0xa2d9a204, 0xa2d9a204, 0xa3d8a304, 0xa4d7a403, 0xa5d7a503,
    0xa5d6a503, 0xa6d6a603, 0xa7d5a703, 0xa7d5a703, 0xa8d4a802, 0xa9d3a902, 0xaad3aa02, 0xaad2aa02, 0xabd2ab02, 0xacd1ac02,
    0xadd0ad02, 0xadd0ad01, 0xaecfae01, 0xafcfaf01, 0xb0ceb001, 0xb0cdb001, 0xb1cdb101, 0xb2ccb201, 0xb2cbb201, 0xb3cbb301,
    0xb4cab401, 0xb5cab501, 0xb5c9b500, 0xb6c8b600, 0xb7c8b700, 0xb7c7b700, 0xb8c6b800, 0xb9c6b900, 0xbac5ba00, 0xbac4ba00,
    0xbbc4bb00, 0xbcc3bc00, 0xbcc2bc00, 0xbdc2bd00, 0xbec1be00, 0xbec0be00, 0xbfc0bf00, 0xc0bfc000, 0xc0bec000, 0xc1bec100,
    0xc2bdc200, 0xc2bcc200, 0xc3bcc300, 0xc4bbc400, 0xc4bac400, 0xc5bac500, 0xc6b9c600, 0xc6b8c600, 0xc7b7c700, 0xc8b7c800,
    0xc8b6c800, 0xc9b5c901, 0xcab5ca01, 0xcab4ca01, 0xcbb3cb01, 0xcbb2cb01, 0xccb2cc01, 0xcdb1cd01, 0xcdb0cd01, 0xceb0ce01,
    0xcfafcf01, 0xcfaecf01, 0xd0add002, 0xd0add002, 0xd1acd102, 0xd2abd202, 0xd2aad202, 0xd3aad302, 0xd3a9d302, 0xd4a8d403,
    0xd5a7d503, 0xd5a7d503, 0xd6a6d603, 0xd6a5d603, 0xd7a5d703, 0xd7a4d704, 0xd8a3d804, 0xd9a2d904, 0xd9a2d904, 0xdaa1da04,
    0xdaa0da05, 0xdb9fdb05, 0xdb9edb05, 0xdc9edc05, 0xdc9ddc05, 0xdd9cdd06, 0xdd9bdd06, 0xde9bde06, 0xde9ade06, 0xdf99df07,
    0xe098e007, 0xe098e007, 0xe197e107, 0xe196e108, 0xe295e208, 0xe295e208, 0xe394e309, 0xe393e309, 0xe492e409, 0xe491e409,
    0xe491e40a, 0xe590e50a, 0xe58fe50a, 0xe68ee60b, 0xe68ee60b, 0xe78de70b, 0xe78ce70c, 0xe88be80c, 0xe88ae80c, 0xe98ae90d,
    0xe989e90d, 0xea88ea0d, 0xea87ea0e, 0xea87ea0e, 0xeb86eb0e, 0xeb85eb0f, 0xec84ec0f, 0xec83ec0f, 0xec83ec10, 0xed82ed10,
    0xed81ed11, 0xee80ee11, 0xee80ee11, 0xee7fee12, 0xef7eef12, 0xef7def13, 0xf07cf013, 0xf07cf013, 0xf07bf014, 0xf17af114,
    0xf179f115, 0xf178f115, 0xf278f215, 0xf277f216, 0xf276f216, 0xf375f317, 0xf375f317, 0xf374f318, 0xf473f418, 0xf472f419,
    0xf471f419, 0xf571f51a, 0xf570f51a, 0xf56ff51b, 0xf66ef61b, 0xf66ef61b, 0xf66df61c, 0xf66cf61c, 0xf76bf71d, 0xf76af71d,
    0xf76af71e, 0xf869f81e, 0xf868f81f, 0xf867f81f, 0xf867f820, 0xf966f921, 0xf965f921, 0xf964f922, 0xf964f922, 0xfa63fa23,
    0xfa62fa23, 0xfa61fa24, 0xfa61fa24, 0xfa60fa25, 0xfb5ffb25, 0xfb5efb26, 0xfb5dfb26, 0xfb5dfb27, 0xfb5cfb28, 0xfc5bfc28,
    0xfc5afc29, 0xfc5afc29, 0xfc59fc2a, 0xfc58fc2a, 0xfc58fc2b, 0xfd57fd2c, 0xfd56fd2c, 0xfd55fd2d, 0xfd55fd2d, 0xfd54fd2e,
    0xfd53fd2f, 0xfd52fd2f, 0xfe52fe30, 0xfe51fe30, 0xfe50fe31, 0xfe4ffe32, 0xfe4ffe32, 0xfe4efe33, 0xfe4dfe34, 0xfe4dfe34,
    0xfe4cfe35, 0xfe4bfe35, 0xfe4afe36, 0xff4aff37, 0xff49ff37, 0xff48ff38, 0xff48ff39, 0xff47ff39, 0xff46ff3a, 0xff45ff3b,
    0xff45ff3b, 0xff44ff3c, 0xff43ff3d, 0xff43ff3d, 0xff42ff3e, 0xff41ff3f, 0xff41ff3f, 0xff40ff40, 0xff3fff41, 0xff3fff41,
    0xff3eff42, 0xff3dff43, 0xff3dff43, 0xff3cff44, 0xff3bff45, 0xff3bff45, 0xff3aff46, 0xff39ff47, 0xff39ff48, 0xff38ff48,
    0xff37ff49, 0xff37ff4a, 0xfe36fe4a, 0xfe35fe4b, 0xfe35fe4c, 0xfe34fe4d, 0xfe34fe4d, 0xfe33fe4e, 0xfe32fe4f, 0xfe32fe4f,
    0xfe31fe50, 0xfe30fe51, 0xfe30fe52, 0xfd2ffd52, 0xfd2ffd53, 0xfd2efd54, 0xfd2dfd55, 0xfd2dfd55, 0xfd2cfd56, 0xfd2cfd57,
    0xfc2bfc58, 0xfc2afc58, 0xfc2afc59, 0xfc29fc5a, 0xfc29fc5a, 0xfc28fc5b, 0xfb28fb5c, 0xfb27fb5d, 0xfb26fb5d, 0xfb26fb5e,
    0xfb25fb5f, 0xfa25fa60, 0xfa24fa61, 0xfa24fa61, 0xfa23fa62, 0xfa23fa63, 0xf922f964, 0xf922f964, 0xf921f965, 0xf921f966,
    0xf820f867, 0xf81ff867, 0xf81ff868, 0xf81ef869, 0xf71ef76a, 0xf71df76a, 0xf71df76b, 0xf61cf66c, 0xf61cf66d, 0xf61bf66e,
    0xf61bf66e, 0xf51bf56f, 0xf51af570, 0xf51af571, 0xf419f471, 0xf419f472, 0xf418f473, 0xf318f374, 0xf317f375, 0xf317f375,
    0xf216f276, 0xf216f277, 0xf215f278, 0xf115f178, 0xf115f179, 0xf114f17a, 0xf014f07b, 0xf013f07c, 0xf013f07c, 0xef13ef7d,
    0xef12ef7e, 0xee12ee7f, 0xee11ee80, 0xee11ee80, 0xed11ed81, 0xed10ed82, 0xec10ec83, 0xec0fec83, 0xec0fec84, 0xeb0feb85,
    0xeb0eeb86, 0xea0eea87, 0xea0eea87, 0xea0dea88, 0xe90de989, 0xe90de98a, 0xe80ce88a, 0xe80ce88b, 0xe70ce78c, 0xe70be78d,
    0xe60be68e, 0xe60be68e, 0xe50ae58f, 0xe50ae590, 0xe40ae491, 0xe409e491, 0xe409e492, 0xe309e393, 0xe309e394, 0xe208e295,
    0xe208e295, 0xe108e196, 0xe107e197, 0xe007e098, 0xe007e098, 0xdf07df99, 0xde06de9a, 0xde06de9b, 0xdd06dd9b, 0xdd06dd9c,
    0xdc05dc9d, 0xdc05dc9e, 0xdb05db9e, 0xdb05db9f, 0xda05daa0, 0xda04daa1, 0xd904d9a2, 0xd904d9a2, 0xd804d8a3, 0xd704d7a4,
    0xd703d7a5, 0xd603d6a5, 0xd603d6a6, 0xd503d5a7, 0xd503d5a7, 0xd403d4a8, 0xd302d3a9, 0xd302d3aa, 0xd202d2aa, 0xd202d2ab,
    0xd102d1ac, 0xd002d0ad, 0xd002d0ad, 0xcf01cfae, 0xcf01cfaf, 0xce01ceb0, 0xcd01cdb0, 0xcd01cdb1, 0xcc01ccb2, 0xcb01cbb2,
    0xcb01cbb3, 0xca01cab4, 0xca01cab5, 0xc901c9b5, 0xc800c8b6, 0xc800c8b7, 0xc700c7b7, 0xc600c6b8, 0xc600c6b9, 0xc500c5ba,
    0xc400c4ba, 0xc400c4bb, 0xc300c3bc, 0xc200c2bc, 0xc200c2bd, 0xc100c1be, 0xc000c0be, 0xc000c0bf, 0xbf00bfc0, 0xbe00bec0,
    0xbe00bec1, 0xbd00bdc2, 0xbc00bcc2, 0xbc00bcc3, 0xbb00bbc4, 0xba00bac4, 0xba00bac5, 0xb900b9c6, 0xb800b8c6, 0xb700b7c7,
    0xb700b7c8, 0xb600b6c8, 0xb500b5c9, 0xb501b5ca, 0xb401b4ca, 0xb301b3cb, 0xb201b2cb, 0xb201b2cc, 0xb101b1cd, 0xb001b0cd,
    0xb001b0ce, 0xaf01afcf, 0xae01aecf, 0xad01add0, 0xad02add0, 0xac02acd1, 0xab02abd2, 0xaa02aad2, 0xaa02aad3, 0xa902a9d3,
    0xa802a8d4, 0xa703a7d5, 0xa703a7d5, 0xa603a6d6, 0xa503a5d6, 0xa503a5d7, 0xa403a4d7, 0xa304a3d8, 0xa204a2d9, 0xa204a2d9,
    0xa104a1da, 0xa004a0da, 0x9f059fdb, 0x9e059edb, 0x9e059edc, 0x9d059ddc, 0x9c059cdd, 0x9b069bdd, 0x9b069bde, 0x9a069ade,
    0x990699df, 0x980798e0, 0x980798e0, 0x970797e1, 0x960796e1, 0x950895e2, 0x950895e2, 0x940894e3, 0x930993e3, 0x920992e4,
    0x910991e4, 0x910991e4, 0x900a90e5, 0x8f0a8fe5, 0x8e0a8ee6, 0x8e0b8ee6, 0x8d0b8de7, 0x8c0b8ce7, 0x8b0c8be8, 0x8a0c8ae8,
    0x8a0c8ae9, 0x890d89e9, 0x880d88ea, 0x870d87ea, 0x870e87ea, 0x860e86eb, 0x850e85eb, 0x840f84ec, 0x830f83ec, 0x830f83ec,
    0x821082ed, 0x811081ed, 0x801180ee, 0x7f117fee, 0x7e127eef, 0x7d127def, 0x7c137cf0, 0x7c137cf0, 0x7b137bf0, 0x7a147af1,
    0x791479f1, 0x781578f1, 0x781578f2, 0x771577f2, 0x761676f2, 0x751675f3, 0x751775f3, 0x741774f3, 0x731873f4, 0x721872f4,
    0x711971f4, 0x711971f5, 0x701a70f5, 0x6f1a6ff5, 0x6e1b6ef6, 0x6e1b6ef6, 0x6d1b6df6, 0x6c1c6cf6, 0x6b1c6bf7, 0x6a1d6af7,
    0x6a1d6af7, 0x691e69f8, 0x681e68f8, 0x671f67f8, 0x671f67f8, 0x662066f9, 0x652165f9, 0x642164f9, 0x642264f9, 0x632263fa,
    0x622362fa, 0x612361fa, 0x612461fa, 0x602460fa, 0x5f255ffb, 0x5e255efb, 0x5d265dfb, 0x5d265dfb, 0x5c275cfb, 0x5b285bfc,
    0x5a285afc, 0x5a295afc, 0x592959fc, 0x582a58fc, 0x582a58fc, 0x572b57fd, 0x562c56fd, 0x552c55fd, 0x552d55fd, 0x542d54fd,
    0x532e53fd, 0x522f52fd, 0x522f52fe, 0x513051fe, 0x503050fe, 0x4f314ffe, 0x4f324ffe, 0x4e324efe, 0x4d334dfe, 0x4d344dfe,
    0x4c344cfe, 0x4b354bfe, 0x4a354afe, 0x4a364aff, 0x493749ff, 0x483748ff, 0x483848ff, 0x473947ff, 0x463946ff, 0x453a45ff,
    0x453b45ff, 0x443b44ff, 0x433c43ff, 0x433d43ff, 0x423d42ff, 0x413e41ff, 0x413f41ff, 0x403f40ff, 0x3f403fff, 0x3f413fff,
    0x3e413eff, 0x3d423dff, 0x3d433dff, 0x3c433cff, 0x3b443bff, 0x3b453bff, 0x3a453aff, 0x394639ff, 0x394739ff, 0x384838ff,
    0x374837ff, 0x374937ff, 0x364a36fe, 0x354a35fe, 0x354b35fe, 0x344c34fe, 0x344d34fe, 0x334d33fe, 0x324e32fe, 0x324f32fe,
    0x314f31fe, 0x305030fe, 0x305130fe, 0x2f522ffd, 0x2f522ffd, 0x2e532efd, 0x2d542dfd, 0x2d552dfd, 0x2c552cfd, 0x2c562cfd,
    0x2b572bfc, 0x2a582afc, 0x2a582afc, 0x295929fc, 0x295a29fc, 0x285a28fc, 0x285b28fb, 0x275c27fb, 0x265d26fb, 0x265d26fb,
    0x255e25fb, 0x255f25fa, 0x246024fa, 0x246124fa, 0x236123fa, 0x236223fa, 0x226322f9, 0x226422f9, 0x216421f9, 0x216521f9,
    0x206620f8, 0x1f671ff8, 0x1f671ff8, 0x1e681ef8, 0x1e691ef7, 0x1d6a1df7, 0x1d6a1df7, 0x1c6b1cf6, 0x1c6c1cf6, 0x1b6d1bf6,
    0x1b6e1bf6, 0x1b6e1bf5, 0x1a6f1af5, 0x1a701af5, 0x197119f4, 0x197119f4, 0x187218f4, 0x187318f3, 0x177417f3, 0x177517f3,
    0x167516f2, 0x167616f2, 0x157715f2, 0x157815f1, 0x157815f1, 0x147914f1, 0x147a14f0, 0x137b13f0, 0x137c13f0, 0x137c13ef,
    0x127d12ef, 0x127e12ee, 0x117f11ee, 0x118011ee, 0x118011ed, 0x108110ed, 0x108210ec, 0x0f830fec, 0x0f830fec, 0x0f840feb,
    0x0e850eeb, 0x0e860eea, 0x0e870eea, 0x0d870dea, 0x0d880de9, 0x0d890de9, 0x0c8a0ce8, 0x0c8a0ce8, 0x0c8b0ce7, 0x0b8c0be7,
    0x0b8d0be6, 0x0b8e0be6, 0x0a8e0ae5, 0x0a8f0ae5, 0x0a900ae4, 0x099109e4, 0x099109e4, 0x099209e3, 0x099309e3, 0x089408e2,
    0x089508e2, 0x089508e1, 0x079607e1, 0x079707e0, 0x079807e0, 0x079807df, 0x069906de, 0x069a06de, 0x069b06dd, 0x069b06dd,
    0x059c05dc, 0x059d05dc, 0x059e05db, 0x059e05db, 0x059f05da, 0x04a004da, 0x04a104d9, 0x04a204d9, 0x04a204d8, 0x04a304d7,
    0x03a403d7, 0x03a503d6, 0x03a503d6, 0x03a603d5, 0x03a703d5, 0x03a703d4, 0x02a802d3, 0x02a902d3, 0x02aa02d2, 0x02aa02d2,
    0x02ab02d1, 0x02ac02d0, 0x02ad02d0, 0x01ad01cf, 0x01ae01cf, 0x01af01ce, 0x01b001cd, 0x01b001cd, 0x01b101cc, 0x01b201cb,
    0x01b201cb, 0x01b301ca, 0x01b401ca, 0x01b501c9, 0x00b500c8, 0x00b600c8, 0x00b700c7, 0x00b700c6, 0x00b800c6, 0x00b900c5,
    0x00ba00c4, 0x00ba00c4, 0x00bb00c3, 0x00bc00c2, 0x00bc00c2, 0x00bd00c1, 0x00be00c0, 0x00be00c0, 0x00bf00bf, 0x00c000be,
    0x00c000be, 0x00c100bd, 0x00c200bc, 0x00c200bc, 0x00c300bb, 0x00c400ba, 0x00c400ba, 0x00c500b9, 0x00c600b8, 0x00c600b7,
    0x00c700b7, 0x00c800b6, 0x00c800b5, 0x01c901b5, 0x01ca01b4, 0x01ca01b3, 0x01cb01b2, 0x01cb01b2, 0x01cc01b1, 0x01cd01b0,
    0x01cd01b0, 0x01ce01af, 0x01cf01ae, 0x01cf01ad, 0x02d002ad, 0x02d002ac, 0x02d102ab, 0x02d202aa, 0x02d202aa, 0x02d302a9,
    0x02d302a8, 0x03d403a7, 0x03d503a7, 0x03d503a6, 0x03d603a5, 0x03d603a5, 0x03d703a4, 0x04d704a3, 0x04d804a2, 0x04d904a2,
    0x04d904a1, 0x04da04a0, 0x05da059f, 0x05db059e, 0x05db059e, 0x05dc059d, 0x05dc059c, 0x06dd069b, 0x06dd069b, 0x06de069a,
    0x06de0699, 0x07df0798, 0x07e00798, 0x07e00797, 0x07e10796, 0x08e10895, 0x08e20895, 0x08e20894, 0x09e30993, 0x09e30992,
    0x09e40991, 0x09e40991, 0x0ae40a90, 0x0ae50a8f, 0x0ae50a8e, 0x0be60b8e, 0x0be60b8d, 0x0be70b8c, 0x0ce70c8b, 0x0ce80c8a,
    0x0ce80c8a, 0x0de90d89, 0x0de90d88, 0x0dea0d87, 0x0eea0e87, 0x0eea0e86, 0x0eeb0e85, 0x0feb0f84, 0x0fec0f83, 0x0fec0f83,
    0x10ec1082, 0x10ed1081, 0x11ed1180, 0x11ee1180, 0x11ee117f, 0x12ee127e, 0x12ef127d, 0x13ef137c, 0x13f0137c, 0x13f0137b,
    0x14f0147a, 0x14f11479, 0x15f11578, 0x15f11578, 0x15f21577, 0x16f21676, 0x16f21675, 0x17f31775, 0x17f31774, 0x18f31873,
    0x18f41872, 0x19f41971, 0x19f41971, 0x1af51a70, 0x1af51a6f, 0x1bf51b6e, 0x1bf61b6e, 0x1bf61b6d, 0x1cf61c6c, 0x1cf61c6b,
    0x1df71d6a, 0x1df71d6a, 0x1ef71e69, 0x1ef81e68, 0x1ff81f67, 0x1ff81f67, 0x20f82066, 0x21f92165, 0x21f92164, 0x22f92264,
    0x22f92263, 0x23fa2362, 0x23fa2361, 0x24fa2461, 0x24fa2460, 0x25fa255f, 0x25fb255e, 0x26fb265d, 0x26fb265d, 0x27fb275c,
    0x28fb285b, 0x28fc285a, 0x29fc295a, 0x29fc2959, 0x2afc2a58, 0x2afc2a58, 0x2bfc2b57, 0x2cfd2c56, 0x2cfd2c55, 0x2dfd2d55,
    0x2dfd2d54, 0x2efd2e53, 0x2ffd2f52, 0x2ffd2f52, 0x30fe3051, 0x30fe3050, 0x31fe314f, 0x32fe324f, 0x32fe324e, 0x33fe334d,
    0x34fe344d, 0x34fe344c, 0x35fe354b, 0x35fe354a, 0x36fe364a, 0x37ff3749, 0x37ff3748, 0x38ff3848, 0x39ff3947, 0x39ff3946,
    0x3aff3a45, 0x3bff3b45, 0x3bff3b44, 0x3cff3c43, 0x3dff3d43, 0x3dff3d42, 0x3eff3e41, 0x3fff3f41, 0x3fff3f40, 0x40ff403f,
    0x41ff413f, 0x41ff413e, 0x42ff423d, 0x43ff433d, 0x43ff433c, 0x44ff443b, 0x45ff453b, 0x45ff453a, 0x46ff4639, 0x47ff4739,
    0x48ff4838, 0x48ff4837, 0x49ff4937, 0x4aff4a36, 0x4afe4a35, 0x4bfe4b35, 0x4cfe4c34, 0x4dfe4d34, 0x4dfe4d33, 0x4efe4e32,
    0x4ffe4f32, 0x4ffe4f31, 0x50fe5030, 0x51fe5130, 0x52fe522f, 0x52fd522f, 0x53fd532e, 0x54fd542d, 0x55fd552d, 0x55fd552c,
    0x56fd562c, 0x57fd572b, 0x58fc582a, 0x58fc582a, 0x59fc5929, 0x5afc5a29, 0x5afc5a28, 0x5bfc5b28, 0x5cfb5c27, 0x5dfb5d26,
    0x5dfb5d26, 0x5efb5e25, 0x5ffb5f25, 0x60fa6024, 0x61fa6124, 0x61fa6123, 0x62fa6223, 0x63fa6322, 0x64f96422, 0x64f96421,
    0x65f96521, 0x66f96620, 0x67f8671f, 0x67f8671f, 0x68f8681e, 0x69f8691e, 0x6af76a1d, 0x6af76a1d, 0x6bf76b1c, 0x6cf66c1c,
    0x6df66d1b, 0x6ef66e1b, 0x6ef66e1b, 0x6ff56f1a, 0x70f5701a, 0x71f57119, 0x71f47119, 0x72f47218, 0x73f47318, 0x74f37417,
    0x75f37517, 0x75f37516, 0x76f27616, 0x77f27715, 0x78f27815, 0x78f17815, 0x79f17914, 0x7af17a14, 0x7bf07b13, 0x7cf07c13,
    0x7cf07c13, 0x7def7d12, 0x7eef7e12, 0x00000000
};

unsigned
make_color(int i, int max_color)
{
    if (i >= max_color - 1)
        return 0;      // black
    i *= 1024;
    i /= max_color - 2;
    i += 1;
    return all_colors[i];
}

//=== main program =======================================================
#define WIDTH  (512*16)
#define HEIGHT (512*16)

static uint16_t __attribute__ ((aligned(64))) image[WIDTH * HEIGHT];

void
help(char *progname)
{

    puts("SSE fractal generator (compiled 64-bit version)");
    puts("");

    printf("%s -p procedure -w width -h height -xmin v.w -xmax v.w -ymin v.w -ymax v.w -t v.w -i maxiter\n", progname);
    puts("Parameters:");
    puts("");
    puts("-p");
    puts("ORIG - select unmodified naive procedure");
    puts("FPU - select FPU procedure (default)");
    puts("SSE - select SSE4.1 procedure");
#if defined(AVX2)
    puts("AVX2 - select AVX2 procedure");
#if defined(FMA)
    puts("AVX2+FMA - select AVX2+FMA procedure");
    puts("AVX2+FMA+STITCH - select AVX2+FMA procedure with code stitching");
#endif
#endif
#if defined(AVX512)
    puts("AVX512 - select AVX512 procedure");
#if defined(FMA)
    puts("AVX512+FMA - select AVX512 using FMA instructions");
    puts("AVX512+FMA+STITCH - select AVX512+FMA procedure with code stitching");
#endif
#endif
    puts("-xmin Remin -ymin Immin -xmax Remax -ymax Immax - define area of calculations; default -2.0 -2.0 +2.0 +2.0");
    puts("-t threshold - define max radius, greater than 0; default 20.0");
    puts("-i maxiters  - define max number of iterations; default 255");
    puts("-xpm - generate xpm format (colours)");
    puts("-pgm - generate pgm format (grey scale)");
    exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    FILE *f = NULL;

    int i, j;
    uint32_t t1, t2;
    void (*function) (float Re_min, float Re_max, float Im_min, float Im_max,
                      float threshold, int maxiters, int width, int height, uint16_t * data) = FPU_mandelbrot;

    // parameters
    char image_name[256];
    uint16_t *ptr;
    char *function_name = "FPU";
    unsigned height = 512;
    unsigned width = 512;
    float Re_min = -2.0, Re_max = +2.0;
    float Im_min = -2.0, Im_max = +2.0;
    float threshold = 20.0;
    unsigned maxiters = 255;
    unsigned xpm = 0;
    unsigned pgm = 0;

    if (argc == 1) {
        help(argv[0]);
    }

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) {
            char *str = argv[++i];

            function_name = str;

            // 1. function name
            if (strcasecmp(str, "ORIG") == 0)
                function = ORIG_mandelbrot;

            else if (strcasecmp(str, "FPU") == 0)
                function = FPU_mandelbrot;

#if defined(SSE4)
            else if (strcasecmp(str, "SSE") == 0)
                function = SSE_mandelbrot;
#endif

#if defined(AVX2)
            else if (strcasecmp(str, "AVX2") == 0)
                function = AVX2_mandelbrot;
#if defined(FMA)
            else if (strcasecmp(str, "AVX2+FMA") == 0)
                function = AVX2_FMA_mandelbrot;
            else if (strcasecmp(str, "AVX2+FMA+STITCH") == 0)
                function = AVX2_FMA_STITCH_mandelbrot;
#endif
#endif
#if defined(AVX512)
            else if (strcasecmp(str, "AVX512") == 0)
                function = AVX512_mandelbrot;
#if defined(FMA)
            else if (strcasecmp(str, "AVX512+FMA") == 0)
                function = AVX512_FMA_mandelbrot;
            else if (strcasecmp(str, "AVX512+FMA+STITCH") == 0)
                function = AVX512_FMA_STITCH_mandelbrot;
#endif
#endif
            else
                help(argv[0]);

            continue;
        }

        if (!strcmp(argv[i], "-w")) {
            width = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-h")) {
            height = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-xmin")) {
            Re_min = atof(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-xmax")) {
            Re_max = atof(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-ymin")) {
            Im_min = atof(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-ymax")) {
            Im_max = atof(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "-t")) {
            threshold = atof(argv[++i]);
            continue;
        }

        if (!strcmp(argv[i], "-i")) {
            maxiters = atoi(argv[++i]);
            continue;
        }

        if (!strcmp(argv[i], "-xpm")) {
            xpm = 1;
            continue;
        }

        if (!strcmp(argv[i], "-pgm")) {
            pgm = 1;
            continue;
        }

        printf("%s ????\n", argv[i]);
        die("unknown parameter on command line");
    }

    if (width * height > WIDTH * HEIGHT) {
        die("width*height (-w, -h) must be less than 8192*8192");
    }
    if (width % 16) {
        die("width (-w) must be a multiple of 16");
    }
    if (height % 16) {
        die("height (-h) must be a multiple of 16");
    }
    if (Re_min >= Re_max) {
        die("wrong window definition (-xmin, -xmax)");
    }
    if (Im_min >= Im_max) {
        die("wrong window definition (-ymin, -ymax)");
    }
    if (threshold <= 1) {
        die("threshold (-t) must be greater than 1");
    }

    // print summary
    printf("Image %d x %d, Area [(%0.5f,%0.5f), (%0.5f, %0.5f)], threshold=%0.2f, maxiters=%d\n",
           width, height, Re_min, Im_min, Re_max, Im_max, threshold, maxiters);

    printf("%s ", function_name);
    fflush(stdout);
    t1 = get_time();
    function(Re_min, Re_max, Im_min, Im_max, threshold, maxiters, width, height, image);
    t2 = get_time();
    printf("%d us\n", t2 - t1);

    if (xpm || pgm) {
        unsigned miniters = maxiters + 2;
        maxiters = 0;
        ptr = image;
        i = height * width;
        while (i--) {
           unsigned pix = *ptr++;
           maxiters = pix > maxiters ? pix : maxiters;   
           miniters = pix < miniters ? pix : miniters;   
	}

        if (xpm) {
            int maxcolors = 512;

            // save xpm image
            sprintf(image_name, "%s.xpm", function_name);
            f = fopen(image_name, "wt");
            if (f) {
                fprintf(f, "/* XPM */\nstatic char * XFACE[] = {\n\"%u %u %u 2\",\n", width, height, maxcolors);
                i = maxcolors;
                while (i--) {
		unsigned tt = make_color(i, maxcolors);
                int b = (tt >> 0) & 0xff;
                int g = (tt >> 8) & 0xff;
                int r = (tt >> 16) & 0xff;
		    char c1 = 'a' + (i / 25);
                    char c2 = 'a' + (i % 25);
                    fprintf(f, "\"%c%c c #%2.2x%2.2x%2.2x\",\n", c1, c2, r, g, b);
                }
                ptr = image;
                i = height;
                while (i--) {
                    j = width;
                    fprintf(f, "\"");
                    while (j--) {
                        double pixel = (double) (*ptr++ - miniters) / (double) (maxiters  - miniters + 1) * (double) maxcolors;
                        int color = (int) pixel;
                        char c1 = 'a' + (color / 25);
                        char c2 = 'a' + (color % 25);

                        fprintf(f, "%c%c", c1, c2);
                    }
                    fprintf(f, "\",\n");
                }
                fprintf(f, "};\n");
                fclose(f);
            }
        }

        if (pgm)
        {
            // save pgm image
            sprintf(image_name, "%s.pgm", function_name);
            f = fopen(image_name, "wb");
            if (f) {
                fprintf(f, "P5\n%d %d\n255\n", width, height);
                ptr = image;
                i = height;
                while (i--) {
                    j = width;
                    while (j--) {
                        double pixel = (double) (*ptr++ - miniters) / (double) (maxiters - miniters + 1) * (double) 255;
                        char color = (char) pixel;
                        fwrite(&color, 1, 1, f);
		    }
                }
                fclose(f);
            }
        }
    }
    return 0;
}
