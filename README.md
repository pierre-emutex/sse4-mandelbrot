# Mandelbrot speed updates over SIMD versions

## Original project

This project is taken from https://github.com/WojciechMula/toys/tree/master/sse4-mandelbrot

> Mandelbrot fractal generator --- SSE2, SSE4, AVX2 and AVX512 implementations

All credits and rights remain to the original author. 

## Original copyright

```
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
*/
```

## Speed-up

The original author mentions:

> Average speedup over FPU procedure is around 4.5 times

This is not quite true, speedup is much, much, much larger for the following reasons

- GCC default compilation uses "-march=native", which creates very very different code already leveraging SIMD floating point instructions when float/double types are used. e.g. SSE4 implementations must be compiled with AVX,AVX2,AVX256 explicitly disabled.

- Original code is not the most efficient, there are simple C simplifications which makes it already faster. Taking care of trivial pipelining is quite important. GCC is able to unroll code when loop invariants do not overlab with start and end conditions. 

- Fine grain tuning can be made with Intel IACA, which exposes execution latencies and bottlenecks (https://software.intel.com/en-us/articles/intel-architecture-code-analyzer). Latencies and Troughput computed from this tool show the potential gain that code stitching would help to achieve (interleave similar code for independent data flow).

- Code scales with openMP (as do most of mandelbrot implementation, the problem to solve is a paradigm of parallelism) but scales even over hyperthreads, which shows the internal execution ports in the CPU are not saturated by a single hyperthread (https://en.wikipedia.org/wiki/Hyper-threading).

- Some procesors (https://software.intel.com/en-us/forums/intel-isa-extensions/topic/737959 and https://ark.intel.com) do have 2 fma (fused multiply and add) units, this nearly helps to  double the throughput, since fma instructions are the bottleneck.

**In fact, speed-up can be made much larger than 80.**

# Updated measurements

## SW updates

Heavily modified by Pierre, 20-05-2018

```
SSE4 computes 4 pixels in parallel
AVX2 computes 8 pixels in parallel
AVX512 computes 16 pixels in parallel
STITCH stitched code computes 2 strands of 16 pixels in parallel
OPENMP execution on hyperthreads computes 4 strands of 16 pixels in parallel on the same core

Improve grey and color rendering, and images can be compared and verified
```

## Compile and Run

```
make example
```

## Results : execution time

The following execution times are taken on an "Intel(R) i7-8650U" processor for the same part of the mandelbrot set.

### gcc 7.3 time (secs)

```
Original C code (ORIG) ................... :       164.
Simplified C code (FPU) .................. :       137.
SSE4 code (SSE4) ......................... :        26.0
AVX2 code (AVX2) ......................... :        10.3
AVX2,FMA code (AVX2+FMA) ................. :         9.45
AVX2,FMA code (AVX2+FMA+STITCH) .......... :         4.93
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         2.73     (2 different cores)
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         2.00     (4 hyperthreads on 2 different cores)
AVX512,FMA,OPENMP with 2 FMA units ....... :         1.20     (estimated)
```

### clang 6.1 time (secs)

```
Original C code (ORIG) ................... :       148.
Simplified C code (FPU) .................. :       137.
SSE4 code (SSE4) ......................... :        31.3
AVX2 code (AVX2) ......................... :        17.2
AVX2,FMA code (AVX2+FMA) ................. :        11.4
AVX2,FMA code (AVX2+FMA+STITCH) .......... :         3.89
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         --- 
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         --- 
AVX512,FMA,OPENMP with 2 FMA units ....... :         --- 
```

