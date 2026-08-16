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
__global__ void gpuMemsetImpleKernel(T *dst, const T value, const size_t size)
{
    const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if (index < size)
    {
        dst[index] = value;
    }
}


void gpuMemsetKernel(void *dst, const void *value, const ScalarType dst_stype, const ScalarType value_stype, const size_t size)
{
    constexpr size_t memset_block_thread_t = 512;
    dim3 grid_set = cudaExecGrid(size, memset_block_thread_t);

    switch (dst_stype)
    {
        case ScalarType::UInt8:     
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt8>::type>(value, value_stype), size);        
            break;
        case ScalarType::UInt16: 
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt16>::type>(value, value_stype), size);   
            break;
        case ScalarType::UInt32: 
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt32>::type>(value, value_stype), size);  
            break;
        case ScalarType::UInt64:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::UInt64>::type>(value, value_stype), size);  
            break;
        case ScalarType::Int8:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int8>::type>(value, value_stype), size);  
            break;
        case ScalarType::Int16:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int16>::type>(value, value_stype), size);
            break;
        case ScalarType::Int32:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int32>::type>(value, value_stype), size);
            break;
        case ScalarType::Int64:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Int64>::type>(value, value_stype), size);
            break;
    #ifdef HALF
        case ScalarType::Float16:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float16>::type>(value, value_stype), size);
            break;
    #endif
        case ScalarType::Float32:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float32>::type>(value, value_stype), size);
            break;
        case ScalarType::Float64:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Float64>::type>(value, value_stype), size);
            break;
        case ScalarType::Bool:
            gpuMemsetImpleKernel<<<grid_set, memset_block_thread_t>>>(static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), 
                                                                       cvtValue<ScalarTypeToCppType<ScalarType::Bool>::type>(value, value_stype), size);
            break;
    }

    cudaErrorChecker(cudaPeekAtLastError(), "GpuDataProc.cu", "ndpp::ndpp_memory::ndpp_cuda::gpuMemsetKernel()");
}



template<typename T1, typename T2>
__global__ void gpuCvtToScalarTypeImpleKernel(const T1 *src, T2 *dst, const size_t size)
{
    const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if (index < size)
    {
        dst[index] = cvtValue<T2>(&src[index], CppTypeToScalarType<T1>::value);
    }
}


template<typename T> 
static inline void gpuCvtToScalarTypeDstTypeKernel(const T *src, void *dst, const ScalarType dst_stype, const size_t size)
{
    constexpr size_t astype_block_thread_t = 512;
    dim3 grid_set = cudaExecGrid(size, astype_block_thread_t);

    switch (dst_stype)
    {
        case ScalarType::UInt8:     
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), size);        
            break;
        case ScalarType::UInt16: 
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), size);  
            break;
        case ScalarType::UInt32: 
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), size);    
            break;
        case ScalarType::UInt64:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), size);
            break;
        case ScalarType::Int8:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), size);
            break;
        case ScalarType::Int16:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), size);
            break;
        case ScalarType::Int32:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), size);
            break;
        case ScalarType::Int64:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), size);
            break;
    #ifdef HALF
        case ScalarType::Float16:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), size);
            break;
    #endif
        case ScalarType::Float32:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), size);
            break;
        case ScalarType::Float64:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), size);
            break;
        case ScalarType::Bool:
            gpuCvtToScalarTypeImpleKernel<<<grid_set, astype_block_thread_t>>>(src, static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), size);
            break;
    }
}


void gpuCvtScalarTypeKernel(const void *src, const ScalarType src_stype, 
                            void *dst, const ScalarType dst_stype, const size_t size)
{
    switch (src_stype)
    {
        case ScalarType::UInt8:     
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt8>::type*>(src), dst, dst_stype, size);           
            break;
        case ScalarType::UInt16: 
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt16>::type*>(src), dst, dst_stype, size);
            break;
        case ScalarType::UInt32: 
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt32>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::UInt64:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt64>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Int8:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int8>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Int16:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int16>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Int32:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int32>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Int64:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int64>::type*>(src), dst, dst_stype, size);  
            break;
    #ifdef HALF
        case ScalarType::Float16:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float16>::type*>(src), dst, dst_stype, size);  
            break;
    #endif
        case ScalarType::Float32:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float32>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Float64:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float64>::type*>(src), dst, dst_stype, size);  
            break;
        case ScalarType::Bool:
            gpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Bool>::type*>(src), dst, dst_stype, size);  
            break;
    }

    cudaErrorChecker(cudaPeekAtLastError(), "GpuDataProc.cu", "ndpp::ndpp_memory::ndpp_cuda::gpuCvtScalarTypeKernel()");
}


}; // namespace ndpp::ndpp_memory::ndpp_cuda

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp