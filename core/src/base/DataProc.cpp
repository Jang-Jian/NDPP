#include <include/base/DataProc.hpp>
#ifdef CUDA
#include <include/base/GpuDataProc.cuh>
#endif


namespace ndpp
{

namespace ndpp_memory
{

// cpuMemsetImpleKernel(OpenMP): Specifying the value at specific memory address.
template<typename T>
static inline void cpuMemsetImpleKernel(T *dst, const T value, const ScalarType value_stype, const size_t size)
{
    #pragma omp parallel for simd schedule(guided) num_threads(threadsReq(size))
    for (size_t index = 0; index < size; ++index)
    {
        dst[index] = value;
    }
}


static inline void cpuMemsetKernel(void *dst, const void *value, 
                                   const ScalarType dst_stype, const ScalarType value_stype, 
                                   const size_t size)
{
    switch (dst_stype)
    {
        case ScalarType::UInt8:                     
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::UInt8>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::UInt16: 
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::UInt16>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::UInt32: 
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::UInt32>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::UInt64:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::UInt64>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Int8:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Int8>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Int16:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Int16>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Int32:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Int32>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Int64:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Int64>::type>(value, value_stype), value_stype, size);
            break;
    #ifdef HALF
        case ScalarType::Float16:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Float16>::type>(value, value_stype), value_stype, size);
            break;
    #endif                       
        case ScalarType::Float32:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Float32>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Float64:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Float64>::type>(value, value_stype), value_stype, size);
            break;
        case ScalarType::Bool:
            cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), 
                                 cvtValue<ScalarTypeToCppType<ScalarType::Bool>::type>(value, value_stype), value_stype, size);
            break;
    }
}


void memset(void *dst, const void *value, 
            const ScalarType dst_stype, const ScalarType value_stype,
            const DeviceType dst_dtype, const size_t size, 
            const string &file_name, const string &method_name)
{
    if (!dst || !value)
    {
        return;
    }

    switch (dst_dtype)
    {
        case DeviceType::Host:
    #ifdef CUDA
        case DeviceType::CudaPinned:
        case DeviceType::CudaUnified:
        case DeviceType::CudaZeroCpy:
    #endif
            {
                cpuMemsetKernel(dst, value, dst_stype, value_stype, size);
            }
            break;

    #ifdef CUDA
        case DeviceType::CudaDevice:
            {
                ndpp_cuda::gpuMemsetKernel(dst, value, dst_stype, value_stype, size);
            }
            break;
    #endif
    }
}


// cpuCvtToScalarTypeImpleKernel(OpenMP): Deep copy via different data type.
template<typename T1, typename T2>
static inline void cpuCvtToScalarTypeImpleKernel(const T1 *src, T2 *dst, const size_t size)
{
    #pragma omp parallel for simd schedule(guided) num_threads(threadsReq(size))
    for (size_t index = 0; index < size; ++index)
    {
        dst[index] = cvtValue<T2>(&src[index], CppTypeToScalarType<T1>::value);
    }
}


template<typename T> 
static inline void cpuCvtToScalarTypeDstTypeKernel(const T *src, void *dst, const ScalarType dst_stype, const size_t size)
{
    switch (dst_stype)
    {
        case ScalarType::UInt8:     
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::UInt8>::type*>(dst), size);        
            break;
        case ScalarType::UInt16: 
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), size);  
            break;
        case ScalarType::UInt32: 
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), size);    
            break;
        case ScalarType::UInt64:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), size);
            break;
        case ScalarType::Int8:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), size);
            break;
        case ScalarType::Int16:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), size);
            break;
        case ScalarType::Int32:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), size);
            break;
        case ScalarType::Int64:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), size);
            break;
    #ifdef HALF
        case ScalarType::Float16:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), size);
            break;
    #endif
        case ScalarType::Float32:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), size);
            break;
        case ScalarType::Float64:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), size);
            break;
        case ScalarType::Bool:
            cpuCvtToScalarTypeImpleKernel(src, static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), size);
            break;
    }
}


void cvtScalarType(const void *src, const ScalarType src_stype, 
                   void *dst, const ScalarType dst_stype, const DeviceType dtype, const size_t size)
{
    if (!src || !dst)
    {
        return;
    }

    switch (dtype)
    {
        case DeviceType::Host:
    #ifdef CUDA
        case DeviceType::CudaPinned:
        case DeviceType::CudaUnified:
        case DeviceType::CudaZeroCpy:
    #endif
            {
                switch (src_stype)
                {
                    case ScalarType::UInt8:     
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt8>::type*>(src), dst, dst_stype, size);           
                        break;
                    case ScalarType::UInt16: 
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt16>::type*>(src), dst, dst_stype, size);
                        break;
                    case ScalarType::UInt32: 
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt32>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::UInt64:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::UInt64>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Int8:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int8>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Int16:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int16>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Int32:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int32>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Int64:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Int64>::type*>(src), dst, dst_stype, size);  
                        break;
                #ifdef HALF
                    case ScalarType::Float16:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float16>::type*>(src), dst, dst_stype, size);  
                        break;
                #endif
                    case ScalarType::Float32:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float32>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Float64:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Float64>::type*>(src), dst, dst_stype, size);  
                        break;
                    case ScalarType::Bool:
                        cpuCvtToScalarTypeDstTypeKernel(static_cast<const ScalarTypeToCppType<ScalarType::Bool>::type*>(src), dst, dst_stype, size);  
                        break;
                }
            }
            break;

    #ifdef CUDA
        case DeviceType::CudaDevice:
            {
                ndpp_cuda::gpuCvtScalarTypeKernel(src, src_stype, dst, dst_stype, size);
            }
            break;
    #endif
    }
}


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp