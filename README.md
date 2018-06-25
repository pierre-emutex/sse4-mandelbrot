# Mandelbrot speed updates over SIMD versions

## Original project

This project is taken from https://github.com/WojciechMula/toys/tree/master/sse4-mandelbrot

```
Mandelbrot fractal generator --- SSE2, SSE4, AVX2 and AVX512 implementations
```

All credits and rights remain to the original author. 

## Speed-up

The original author mentions:

> Average speedup over FPU procedure is around 4.5 times

This is not quite true, speedup is much, much, much larger for the following reasons

- default compilation use "-march=native", which creates very very different code already leveraging SIMD floating point instructions when float/double types are used. e.g. SSE4 implementations must be compiled with AVX,AVX2,AVX256 explicitly disabled.

- original code is not the most efficient, there are simple C simplifications which makes it already faster. Taking care of trivial pipelining is quite important. GCC is able to unroll code when loop invariants do not overlab with start and end conditions. 

- fine grain tuning can be made with Intel IACA, which exposes execution latencies and bottlenecks
https://software.intel.com/en-us/articles/intel-architecture-code-analyzer
Latencies and troghput computed friom this tool show the differences between latencies and throughput, and the potential gain that code stitching would help to achieve.

- some procesors (https://software.intel.com/en-us/forums/intel-isa-extensions/topic/737959 and https://ark.intel.com)do have 2 fma (fused multiply and add) units, this nearly helps to  double the throughput, since fma instructions are the bottleneck

- code scales with openMP (as do most of mandelbrot implementation, the problem to solve is a paradigm of parallelism) but scales even over hyperthreads, which shows the internal execution ports in the CPU are not saturated by 1 hyperthread.


**In fact Measured speed-up is much larger than 80.**

# Updated measurements

## Compile and Run

```
make run
```

## Execution time

The following execution times are taken on an "Intel(R) i7-8650U" processor for the same part of the mandelbrot set.

```
                                                   time (secs)
Original C code (ORIG) ................... :       164
Simplified C code (FPU) .................. :       137
SSE4 code (SSE4) ......................... :        26.0
AVX2 code (AVX2) ......................... :        10.3
AVX2,FMA code (AVX2+FMA) ................. :         9.45
AVX2,FMA code (AVX2+FMA+STITCH) .......... :         4.93
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         2.73     (2 different cores)
AVX2,FMA,OPENMP code (AVX2+FMA+STITCH) ... :         2.00     (4 hyperthreads on 2 different cores)
AVX512,FMA,OPENMP with 2 FMA units ....... :         1.20     (estimated)
```


