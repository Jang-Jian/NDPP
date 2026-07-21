#pragma once

#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif


// Used for forcing inline for host & device.
#ifdef CUDA
#define ndppInline __host__ __device__ __forceinline__
#else
#define ndppInline inline __attribute__((always_inline))
#endif