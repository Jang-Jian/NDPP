#include <include/base/DataProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/base/GpuDataProc.cuh>
#include <include/extension/CudaProc.hpp>

#define block_thread_t 512


namespace ndpp
{

namespace ndpp_memory
{

namespace ndpp_cuda
{

template<typename T>
__global__ void gpuMemsetGlobalKernel(T *dst, const void *value, const ScalarType value_stype, const size_t size)
{
    size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const auto fill = [&](const T &v)
        {
            dst[index] = v;
        };

        switch (value_stype)
        {
            case ScalarType::UInt8:  
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt8>::type*>(value)));
                break;
            case ScalarType::UInt16: 
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt16>::type*>(value)));
                break;
            case ScalarType::UInt32: 
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt32>::type*>(value)));
                break;
            case ScalarType::UInt64:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt64>::type*>(value)));
                break;
            case ScalarType::Int8:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Int8>::type*>(value)));
                break;
            case ScalarType::Int16:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Int16>::type*>(value)));
                break;
            case ScalarType::Int32:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Int32>::type*>(value)));
                break;
            case ScalarType::Int64:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Int64>::type*>(value)));
                break;

        #ifdef HALF
            case ScalarType::Float16:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Float16>::type*>(value)));
                break;
        #endif
                
            case ScalarType::Float32:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Float32>::type*>(value)));
                break;
            case ScalarType::Float64:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Float64>::type*>(value)));
                break;
            case ScalarType::Bool:
                fill(static_cast<T>(*static_cast<const ScalarTypeToCppType<ScalarType::Bool>::type*>(value)));
                break;
        }
    }
}


void gpuMemsetImpleKernel(void *dst, const void *value, const ScalarType dst_stype, const ScalarType value_stype, const size_t size)
{
    dim3 grid_set = cudaExecGrid(size, block_thread_t);

    switch (dst_stype)
    {
        case ScalarType::UInt8:     
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), value, value_stype, size);        
            break;
        case ScalarType::UInt16: 
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), value, value_stype, size);   
            break;
        case ScalarType::UInt32: 
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), value, value_stype, size);  
            break;
        case ScalarType::UInt64:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), value, value_stype, size);  
            break;
        case ScalarType::Int8:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), value, value_stype, size);  
            break;
        case ScalarType::Int16:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), value, value_stype, size);
            break;
        case ScalarType::Int32:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), value, value_stype, size);
            break;
        case ScalarType::Int64:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), value, value_stype, size);
            break;

    #ifdef HALF
        case ScalarType::Float16:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), value, value_stype, size);
            break;
    #endif
            
        case ScalarType::Float32:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), value, value_stype, size);
            break;
        case ScalarType::Float64:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), value, value_stype, size);
            break;
        case ScalarType::Bool:
            gpuMemsetGlobalKernel<<<grid_set, block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), value, value_stype, size);
            break;
    }

    cudaErrorChecker(cudaPeekAtLastError(), "GpuDataProc.cu", "ndpp::ndpp_memory::ndpp_cuda::gpuMemsetImpleKernel()");
}

}; // namespace ndpp::ndpp_memory::ndpp_cuda

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp