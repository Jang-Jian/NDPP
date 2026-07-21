#include <include/base/DataProc.hpp>
#include <include/base/DataProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/base/GpuDataProc.cuh>
#include <include/extension/CudaProc.hpp>


namespace ndpp
{

namespace ndpp_memory
{

namespace ndpp_cuda
{

template<typename T>
__global__ void gpuMemsetGlobalKernel(T *dst, const T value, const size_t size)
{
    const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if (index < size)
    {
        dst[index] = value;
    }
}


void gpuMemsetImpleKernel(void *dst, const void *value, const ScalarType dst_stype, const ScalarType value_stype, const size_t size)
{
    constexpr size_t memset_block_thread_t = 512;
    dim3 grid_set = cudaExecGrid(size, memset_block_thread_t);

    switch (dst_stype)
    {
        case ScalarType::UInt8:     
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt8>::type>(value, value_stype), size);        
            break;
        case ScalarType::UInt16: 
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt16>::type>(value, value_stype), size);   
            break;
        case ScalarType::UInt32: 
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt32>::type>(value, value_stype), size);  
            break;
        case ScalarType::UInt64:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt64>::type>(value, value_stype), size);  
            break;
        case ScalarType::Int8:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int8>::type>(value, value_stype), size);  
            break;
        case ScalarType::Int16:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int16>::type>(value, value_stype), size);
            break;
        case ScalarType::Int32:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int32>::type>(value, value_stype), size);
            break;
        case ScalarType::Int64:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int64>::type>(value, value_stype), size);
            break;
    #ifdef HALF
        case ScalarType::Float16:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float16>::type>(value, value_stype), size);
            break;
    #endif
        case ScalarType::Float32:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float32>::type>(value, value_stype), size);
            break;
        case ScalarType::Float64:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float64>::type>(value, value_stype), size);
            break;
        case ScalarType::Bool:
            gpuMemsetGlobalKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Bool>::type>(value, value_stype), size);
            break;
    }

    cudaErrorChecker(cudaPeekAtLastError(), "GpuDataProc.cu", "ndpp::ndpp_memory::ndpp_cuda::gpuMemsetImpleKernel()");
}

}; // namespace ndpp::ndpp_memory::ndpp_cuda

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp