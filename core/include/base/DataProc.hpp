#pragma once

#include <omp.h>

#include <include/base/InlineProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/base/Parallelism.hpp>
#ifdef CUDA
#include <include/base/GpuDataProc.cuh>
#endif


namespace ndpp
{

namespace ndpp_memory
{

// Specifying the value at specific memory address.
inline void memset(void *dst, const void *value, 
                   const ScalarType dst_stype, const ScalarType value_stype,
                   const DeviceType dst_dtype, const size_t size, 
                   const string &file_name, const string &method_name);



template<typename T>
ndppInline T cvtValue(const void *value, const ScalarType value_stype)
{
    switch (value_stype)
    {
        case ScalarType::UInt8:   return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt8>::type*>(value)));
        case ScalarType::UInt16:  return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt16>::type*>(value)));
        case ScalarType::UInt32:  return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt32>::type*>(value)));
        case ScalarType::UInt64:  return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::UInt64>::type*>(value)));
        case ScalarType::Int8:    return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Int8>::type*>(value)));
        case ScalarType::Int16:   return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Int16>::type*>(value)));
        case ScalarType::Int32:   return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Int32>::type*>(value)));
        case ScalarType::Int64:   return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Int64>::type*>(value)));
    #ifdef HALF
        case ScalarType::Float16: return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Float16>::type*>(value)));
    #endif
        case ScalarType::Float32: return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Float32>::type*>(value)));
        case ScalarType::Float64: return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Float64>::type*>(value)));
        case ScalarType::Bool:    return static_cast<T>(static_cast<double>(*static_cast<const ScalarTypeToCppType<ScalarType::Bool>::type*>(value)));
    }

    return 0;
}


// cpuMemsetImpleKernel(OpenMP): Specifying the value at specific memory address.
template<typename T>
inline void cpuMemsetImpleKernel(T *src, const T value, const ScalarType value_stype, const size_t size)
{
    #pragma omp parallel for schedule(guided) num_threads(threadsReq(size))
    for (size_t index = 0; index < size; ++index)
    {
        src[index] = value;
    }
}


inline void memset(void *dst, const void *value, 
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
            break;

    #ifdef CUDA
        case DeviceType::CudaDevice:
            {
                ndpp_cuda::gpuMemsetImpleKernel(dst, value, dst_stype, value_stype, size);
            }
            break;
    #endif
    }

}

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp