#pragma once

#include <omp.h>

#include <include/base/ScalarType.hpp>
#include <include/base/Parallelism.hpp>


namespace ndpp
{

namespace ndpp_memory
{
    static inline void memset(void *dst, const void *value, 
                              const ScalarType dst_stype, const ScalarType value_stype,
                              const DeviceType dst_dtype, const size_t size, 
                              const string &file_name, const string &method_name);

}; // namespace ndpp::ndpp_memory

namespace ndpp_memory
{

template<typename T>
static inline void cpuMemsetImpleKernel(T *src, const void *value, const ScalarType value_stype, const size_t size)
{
    const auto fill = [&](const T &v)
    {
        #pragma omp parallel for schedule(guided) num_threads(threadsReq(size))
        for (size_t index = 0; index < size; ++index)
        {
            src[index] = v;
        }
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


// TO DO ... ADD function 'gpuMemsetImpleKernel' here .
static inline void memset(void *dst, const void *value, 
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
                                             value, value_stype, size);
                        break;
                    case ScalarType::UInt16: 
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt16>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::UInt32: 
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt32>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::UInt64:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::UInt64>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Int8:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int8>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Int16:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int16>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Int32:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int32>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Int64:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Int64>::type*>(dst), 
                                             value, value_stype, size);
                        break;

                #ifdef HALF
                    case ScalarType::Float16:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float16>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                #endif
                        
                    case ScalarType::Float32:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float32>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Float64:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Float64>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                    case ScalarType::Bool:
                        cpuMemsetImpleKernel(static_cast<ScalarTypeToCppType<ScalarType::Bool>::type*>(dst), 
                                             value, value_stype, size);
                        break;
                }
            }
            break;

    #ifdef CUDA
        case DeviceType::CudaDevice:

            break;
    #endif
    }

}; // namespace ndpp::ndpp_memory

}

}; // namespace ndpp