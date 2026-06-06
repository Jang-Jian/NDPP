#pragma once

#include <include/base/ScalarType.hpp>
#include <include/extension/CudaProc.hpp>


namespace ndpp
{

namespace ndpp_memory
{

namespace ndpp_cuda
{

// gpuMemsetImpleKernel(CUDA): Specifying the value at specific memory address.
void gpuMemsetImpleKernel(void *dst, const void *value, const ScalarType dst_stype, const ScalarType value_stype, const size_t size);

}; // namespace ndpp::ndpp_memory::ndpp_cuda

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp