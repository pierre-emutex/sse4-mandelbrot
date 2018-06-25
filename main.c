
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

	Estimated elapsed execution time on KabyLake i7-8650U 3.9Ghz  (1 FMA unit)

                                     clang6   gcc7.3 
	ORIG ....................... 156 s    165 s    Naive textbook implementation
	FPU ........................  90 s    137 s    "Correctly written" source code, variables, and loops
	SSE4 ........................ 20 s     26 s    Replace FPU instructions with SSE instructions
	AVX2 ........................ 10 s     10.2 s    Replace SSE instructions with AVX2 instructions
	AVX2+FMA ..................... 6.9 s    9.3 s  Change algorithm to benefit from fma (5 instructions per loop)
	AVX2+FMA+STITCH .............. 3.8 s    4.9 s  Enhance microarchitecture parallelism and fill latencies
	AVX2+FMA+STITCH+openMP ....... --       2.71 s Run on 2 hyperthreads of the same core
	AVX512+FMA+STITCH+openMP ..... --       1.60 s (estimated) Replace AVX2 instructions with AVX512 instructions
	AVX512+FMA+STITCH+openMP ..... --       0.50 s (estimated) run on 4 cores (8 hyperthreads)

	Usage: example of command line

		 ./fractal64 -p ORIG -w 512 -h 512 -xmin -1 -xmax -1 -ymin 1 -ymax 1 -t 4.0 -i 255

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

//=== main program =======================================================
#define WIDTH  (512*16)
#define HEIGHT (512*16)

static uint8_t __attribute__ ((aligned(64))) image[WIDTH * HEIGHT];

void
help(char *progname)
{

    puts("SSE fractal generator (compiled 64-bit version)");
    puts("");

    printf("%s -p procedure -w width -h height -xmin v.w -xmax v.w -ymin v.w -ymax v.w -t v.w -i maxiter -dry-run\n", progname);
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
    puts("-dry-run - do not save any image file (.pgm format)");
    puts("-xpm - generate xpm format");
    exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    FILE *f = NULL;

    int i, j;
    uint32_t t1, t2;
    void (*function) (float Re_min, float Re_max, float Im_min, float Im_max,
                      float threshold, int maxiters, int width, int height, uint8_t * data) = FPU_mandelbrot;

    // parameters
    char image_name[256];
    uint8_t *ptr;
    char *function_name = "FPU";
    unsigned height = 512;
    unsigned width = 512;
    float Re_min = -2.0, Re_max = +2.0;
    float Im_min = -2.0, Im_max = +2.0;
    float threshold = 20.0;
    unsigned maxiters = 255;
    unsigned dry_run = 0;
    unsigned xpm = 0;

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

        if (!strcmp(argv[i], "-dry_run")) {
            dry_run = 1;
            continue;
        }

        if (!strcmp(argv[i], "-xpm")) {
            xpm = 1;
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

    if (!dry_run) {
        if (xpm) {
            int maxcolors = 52;

            // save xpm image
            sprintf(image_name, "%s.xpm", function_name);
            f = fopen(image_name, "wt");
            if (f) {
                fprintf(f, "/* XPM */\nstatic char * XFACE[] = {\n\"%u %u %u 1\",\n", width, height, maxcolors);
                i = maxcolors;
                while (i--) {
                    // map n on the 0..1 interval
                    double t = (double) i / (double) maxcolors;

                    // Use smooth polynomials for r, g, b
                    double tt = 1.0 - t;
                    int r = (int) (9.0 * tt * t * t * t * 255);
                    int g = (int) (15.0 * tt * tt * t * t * 255);
                    int b = (int) (8.5 * tt * tt * tt * t * 255);
                    char c = (i > 25 ? 'a' - 26 : 'A') + i;

                    fprintf(f, "\"%c c #%2.2x%2.2x%2.2x\",\n", c, r, g, b);
                }
                ptr = image;
                i = height;
                while (i--) {
                    j = width;
                    fprintf(f, "\"");
                    while (j--) {
                        double pixel = (double) *ptr++ / (double) (maxiters + 2) * (double) maxcolors;
                        int color = (int) (pixel);
                        char c = (color > 25 ? 'a' - 26 : 'A') + color;

                        fprintf(f, "%c", c);
                    }
                    fprintf(f, "\",\n");
                }
                fprintf(f, "};\n");
                fclose(f);
            }
        }
        else {
            // save pgm image
            sprintf(image_name, "%s.pgm", function_name);
            f = fopen(image_name, "wb");
            if (f) {
                fprintf(f, "P5\n%d %d\n255\n", width, height);
                fwrite(image, width * height, 1, f);
                fclose(f);
            }
        }
    }
    return 0;
}
