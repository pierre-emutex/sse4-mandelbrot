# Note: flag ``mfpmath=387`` is very important, because in 32-bit version
#       I mixed floating points calculations with asm statements.
FLAGS=-Wall -Wextra -pedantic -O3 -fomit-frame-pointer -fexpensive-optimizations -fno-stack-protector -ffast-math
FORCEFPU=-mfpmath=387
COMPILER=gcc

MAIN=main.c
DEPS=$(MAIN) fpu-proc.c
ALL= \
    fractal64fpu \
    fractal64sse4 \
    fractal64avx2 \
    fractal64avx2fma \
    fractal64avx2fmaopenmp \
    fractal64avx512 \
    fractal64avx512fma \
    fractal64avx512fmaopenmp 

all: $(ALL)

fractal64fpu: $(DEPS)
	$(COMPILER) $(FLAGS) -march=westmere -mno-sse4.2 $(MAIN) -o $@

fractal64sse4: $(DEPS) sse4-proc-64-bit.c
	$(COMPILER) $(FLAGS) -msse4.2 -DSSE4 -march=westmere -mno-avx $(MAIN) -o $@

fractal64avx2: $(DEPS) avx2-proc-64-bit.c
	$(COMPILER) $(FLAGS) -mavx2 -DSSE4 -DAVX2 -march=broadwell -mno-fma -mno-avx512f $(MAIN) -o $@

fractal64avx2fma: $(DEPS) avx2-proc-64-bit.c
	$(COMPILER) $(FLAGS) -mfma -mavx2 -DSSE4 -DAVX2 -DFMA -march=broadwell -mno-avx512f $(MAIN) -o $@

fractal64avx2fmaopenmp: $(DEPS) avx2-proc-64-bit.c
	$(COMPILER) $(FLAGS) -fopenmp -mfma -mavx2 -DSSE4 -DAVX2 -DFMA -march=skylake -mno-avx512f $(MAIN) -o $@

fractal64avx512: $(DEPS) avx2-proc-64-bit.c avx512-proc-64-bit.c
	$(COMPILER) $(FLAGS) -mavx2 -mavx512f -DSSE4 -DAVX2 -DAVX512 $(MAIN) -march=knl -o $@

fractal64avx512fma: $(DEPS) avx2-proc-64-bit.c avx512-proc-64-bit.c
	$(COMPILER) $(FLAGS) -mfma -mavx2 -mavx512f -DSSE4 -DAVX2 -DFMA -DAVX512 -march=knl $(MAIN) -o $@

fractal64avx512fmaopenmp: $(DEPS) avx2-proc-64-bit.c avx512-proc-64-bit.c
	$(COMPILER) $(FLAGS) -fopenmp -mfma -mavx2 -mavx512f -DSSE4 -DAVX2 -DFMA -DAVX512 -march=knl $(MAIN) -o $@

RUN_PARAM=-w 8192 -h 8192 -xmin -1 -ymin -1 -xmax 1 -ymax 1 -t 4.0 -i 1024

run: $(ALL)
	 ./fractal64fpu -p ORIG $(RUN_PARAM)
	 ./fractal64fpu -p FPU $(RUN_PARAM)
	 ./fractal64sse4 -p SSE $(RUN_PARAM)
	 ./fractal64avx2 -p AVX2 $(RUN_PARAM)
	 ./fractal64avx2fma -p AVX2+FMA $(RUN_PARAM)
	 ./fractal64avx2fma -p AVX2+FMA+STITCH $(RUN_PARAM)
	 taskset -c 2,6 ./fractal64avx2fmaopenmp -p AVX2+FMA+STITCH $(RUN_PARAM)
	 taskset -c 2,3,6,7 ./fractal64avx2fmaopenmp -p AVX2+FMA+STITCH $(RUN_PARAM)
	 ./fractal64avx512 -p AVX512 $(RUN_PARAM)
	 ./fractal64avx512fma -p AVX512+FMA $(RUN_PARAM)
	 ./fractal64avx512fma -p AVX512+FMA+STITCH $(RUN_PARAM)
	 taskset -c 2,6 ./fractal64avx512fmaopenmp -p AVX512+FMA+STITCH $(RUN_PARAM)
	 taskset -c 2,3,6,7 ./fractal64avx512fmaopenmp -p AVX512+FMA+STITCH $(RUN_PARAM)

clean:
	rm -f $(ALL)
