#pragma once

#include <include/half/half.hpp>
#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif


namespace ndpp
{

#ifdef CUDA
// Using CUDA's __half for float16_t.
using float16_t = __half;
#else
// Using CUDA's half_float::half for float16_t.
using float16_t = half_float::half;
#endif

}; // namespace ndpp