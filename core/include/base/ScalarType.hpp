#pragma once

#include <cstdint>
#include <iostream>

#ifdef HALF
#include <include/half/half.hpp>
#endif
#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif
#include <include/base/DeviceProc.hpp>
#include <include/logging/Logging.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_memory
{

// ScalarType: Scalar(data) type.
// reference: https://github.com/pytorch/pytorch/blob/3d96925b9b42a329b07cd302e08c3990c15a34a6/torch/headeronly/core/ScalarType.h#L1
enum class ScalarType;

// Convert ndpp_memory::ScalarType to C++'s data type.
// Reference: 
// - https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
// - https://github.com/pytorch/pytorch/blob/3d96925b9b42a329b07cd302e08c3990c15a34a6/torch/headeronly/core/ScalarType.h#L288
template <ScalarType T> struct ScalarTypeToCppType;

// Convert C++'s data type to ndpp_memory::ScalarType.
// Reference:
// - https://github.com/pytorch/pytorch/blob/3d96925b9b42a329b07cd302e08c3990c15a34a6/torch/headeronly/core/ScalarType.h#L270
template <typename T> struct CppTypeToScalarType;

// Get sizeof(T) * num.
template<typename T>
static inline size_t sizeOf(const size_t num);

// Access the pointer via ndpp::ndpp_memory::ScalarType & specific position.
static inline void* scalarPtrShift(const void *src, const ScalarType stype, const int64_t pos);

// Get the size of scalar via ndpp::ndpp_memory::ScalarType.
static inline size_t sizeOfScalar(const ScalarType stype, const string &file_name, const string &method_name);

// Mixed memory allocation (host & cuda) via ndpp::ndpp_memory::ScalarType.
static inline void* mixScalarAlloc(const size_t size, const ScalarType stype, const DeviceType device_type,
                                   const string &file_name, const string &method_name);

// Mixed memory deallocation (host & cuda) via ndpp::ndpp_memory::ScalarType.                                   
static inline void mixScalarDeAlloc(void *src, const ScalarType stype, const DeviceType device_type,
                                    const string &file_name, const string &method_name);

// Mixed memory copy (host & cuda) via ndpp::ndpp_memory::ScalarType.      
static inline void mixScalarCopy(const void *src, const DeviceType src_device_type,
                                 void *dst, const DeviceType dst_device_type, const ScalarType stype, const size_t size,
                                 const string &file_name, const string &method_name);

}; // namespace ndpp::ndpp_memory


namespace ndpp_memory
{


enum class ScalarType
{
    // unsigned integer: 8-bits.
    UInt8  = 0,

    // unsigned integer: 16-bits.
    UInt16 = 1,

    // unsigned integer: 32-bits.
    UInt32 = 2,

    // unsigned integer: 64-bits.
    UInt64 = 3,

    // integer: 8-bits.
    Int8 = 4,

    // integer: 16-bits.
    Int16 = 5,

    // integer: 32-bits.
    Int32 = 6,

    // integer: 64-bits.
    Int64 = 7,

#ifdef HALF
    // float(half): 16-bits.
    Float16 = 8,
#endif

    // float: 32-bits.
    Float32 = 9,

    // float(double): 64-bits.
    Float64 = 10,

    // boolean: 64-bits.
    Bool = 11
};


template<> struct ScalarTypeToCppType<ScalarType::UInt8>   { using type = uint8_t;  static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::UInt16>  { using type = uint16_t; static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::UInt32>  { using type = uint32_t; static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::UInt64>  { using type = uint64_t; static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::Int8>    { using type = int8_t;   static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::Int16>   { using type = int16_t;  static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::Int32>   { using type = int32_t;  static type t;         };
template<> struct ScalarTypeToCppType<ScalarType::Int64>   { using type = int64_t;  static type t;         };
#ifdef HALF
#ifdef CUDA
template<> struct ScalarTypeToCppType<ScalarType::Float16> { using type = __half;           static type t; };
#else
template<> struct ScalarTypeToCppType<ScalarType::Float16> { using type = half_float::half; static type t; };
#endif
#endif
template<> struct ScalarTypeToCppType<ScalarType::Float32> { using type = float;   static type t;          };
template<> struct ScalarTypeToCppType<ScalarType::Float64> { using type = double;  static type t;          };
template<> struct ScalarTypeToCppType<ScalarType::Bool>    { using type = bool;    static type t;          };


template<> struct CppTypeToScalarType<uint8_t>  : std::integral_constant<ScalarType, ScalarType::UInt8>  {};
template<> struct CppTypeToScalarType<uint16_t> : std::integral_constant<ScalarType, ScalarType::UInt16> {};
template<> struct CppTypeToScalarType<uint32_t> : std::integral_constant<ScalarType, ScalarType::UInt32> {};
template<> struct CppTypeToScalarType<uint64_t> : std::integral_constant<ScalarType, ScalarType::UInt64> {};
template<> struct CppTypeToScalarType<int8_t>   : std::integral_constant<ScalarType, ScalarType::Int8>   {};
template<> struct CppTypeToScalarType<int16_t>  : std::integral_constant<ScalarType, ScalarType::Int16>  {};
template<> struct CppTypeToScalarType<int32_t>  : std::integral_constant<ScalarType, ScalarType::Int32>  {};
template<> struct CppTypeToScalarType<int64_t>  : std::integral_constant<ScalarType, ScalarType::Int64>  {};
#ifdef HALF
#ifdef CUDA
template<> struct CppTypeToScalarType<__half>           : std::integral_constant<ScalarType, ScalarType::Float16> {};
template<> struct CppTypeToScalarType<half_float::half> : std::integral_constant<ScalarType, ScalarType::Float16> {};
#else
template<> struct CppTypeToScalarType<half_float::half> : std::integral_constant<ScalarType, ScalarType::Float16> {};
#endif
#endif
template<> struct CppTypeToScalarType<float>  : std::integral_constant<ScalarType, ScalarType::Float32> {};
template<> struct CppTypeToScalarType<double> : std::integral_constant<ScalarType, ScalarType::Float64> {};
template<> struct CppTypeToScalarType<bool>   : std::integral_constant<ScalarType, ScalarType::Bool>    {};


template<typename T>
static inline size_t sizeOf(const size_t num)
{
    return num * sizeof(T);
}


static inline void* scalarPtrShift(const void *src, const ScalarType stype, const int64_t pos)
{
    if (!src)
    {
        return nullptr;
    }

    switch (stype)
    {
        case ScalarType::UInt8:
            return (ScalarTypeToCppType<ScalarType::UInt8>::type*)src + pos;
        case ScalarType::UInt16:
            return (ScalarTypeToCppType<ScalarType::UInt16>::type*)src + pos;
        case ScalarType::UInt32:
            return (ScalarTypeToCppType<ScalarType::UInt32>::type*)src + pos;
        case ScalarType::UInt64:
            return (ScalarTypeToCppType<ScalarType::UInt64>::type*)src + pos;
        case ScalarType::Int8:
            return (ScalarTypeToCppType<ScalarType::Int8>::type*)src + pos;
        case ScalarType::Int16:
            return (ScalarTypeToCppType<ScalarType::Int16>::type*)src + pos;
        case ScalarType::Int32:
            return (ScalarTypeToCppType<ScalarType::Int32>::type*)src + pos;
        case ScalarType::Int64:
            return (ScalarTypeToCppType<ScalarType::Int64>::type*)src + pos;

    #ifdef HALF
        case ScalarType::Float16:
            return (ScalarTypeToCppType<ScalarType::Float16>::type*)src + pos;
    #endif
            
        case ScalarType::Float32:
            return (ScalarTypeToCppType<ScalarType::Float32>::type*)src + pos;
        case ScalarType::Float64:
            return (ScalarTypeToCppType<ScalarType::Float64>::type*)src + pos;
        case ScalarType::Bool:
            return (ScalarTypeToCppType<ScalarType::Bool>::type*)src + pos;
    }

    return nullptr;
}


template<typename T>
static inline T scalarPtrAccess(const void *src, const ScalarType stype, const DeviceType dtype, const int64_t pos,
                                const string &file_name, const string &method_name)
{
    if (!src)
    {
        ndpp_log::logger(file_name, method_name, ndpp_log::RuntimeType::ERROR, 
                         "In ndpp::ndpp_memory::scalarPtrAccess(), the src ptr is nullptr.", true);
        exit(EXIT_FAILURE);
    }

    T data = static_cast<T>(0);

    switch (dtype)
    {
        case DeviceType::Host:
    #ifdef CUDA
        case DeviceType::CudaPinned:
        case DeviceType::CudaUnified:
        case DeviceType::CudaZeroCpy:
    #endif
            switch (stype)
            {
                case ScalarType::UInt8:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::UInt8>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::UInt16:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::UInt16>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::UInt32:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::UInt32>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::UInt64:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::UInt64>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Int8:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Int8>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Int16:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Int16>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Int32:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Int32>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Int64:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Int64>::type*)scalarPtrShift(src, stype, pos));
                    break;

            #ifdef HALF
                case ScalarType::Float16:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Float16>::type*)scalarPtrShift(src, stype, pos));
                    break;
            #endif
                    
                case ScalarType::Float32:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Float32>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Float64:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Float64>::type*)scalarPtrShift(src, stype, pos));
                    break;
                case ScalarType::Bool:
                    data = static_cast<T>(*(ScalarTypeToCppType<ScalarType::Bool>::type*)scalarPtrShift(src, stype, pos));
                    break;
            }
            break;
    #ifdef CUDA
        case DeviceType::CudaDevice:
            switch (stype)
            {
                case ScalarType::UInt8:
                    {
                        ScalarTypeToCppType<ScalarType::UInt8>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt8>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::UInt16:
                    {
                        ScalarTypeToCppType<ScalarType::UInt16>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt16>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::UInt32:
                    {
                        ScalarTypeToCppType<ScalarType::UInt32>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt32>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::UInt64:
                    {
                        ScalarTypeToCppType<ScalarType::UInt64>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt64>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Int8:
                    {
                        ScalarTypeToCppType<ScalarType::Int8>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int8>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Int16:
                    {
                        ScalarTypeToCppType<ScalarType::Int16>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int16>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Int32:
                    {
                        ScalarTypeToCppType<ScalarType::Int32>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int32>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Int64:
                    {
                        ScalarTypeToCppType<ScalarType::Int64>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int64>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;

            #ifdef HALF
                case ScalarType::Float16:
                    {
                        ScalarTypeToCppType<ScalarType::Float16>::type _data = 0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float16>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
            #endif
                    
                case ScalarType::Float32:
                    {
                        ScalarTypeToCppType<ScalarType::Float32>::type _data = 0.0f;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float32>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Float64:
                    {
                        ScalarTypeToCppType<ScalarType::Float64>::type _data = 0.0;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float64>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
                case ScalarType::Bool:
                    {
                        ScalarTypeToCppType<ScalarType::Bool>::type _data = false;
                        mixMemoryCopy((ScalarTypeToCppType<ScalarType::Bool>::type*)scalarPtrShift(src, stype, pos), 
                                      DeviceType::CudaDevice, &_data, DeviceType::Host, 1, file_name, method_name);
                        data = static_cast<T>(_data);
                    }
                    break;
            }
            break;
    #endif
    }

    return data;
}


static inline size_t sizeOfScalar(const ScalarType stype, const string &file_name, 
                                  const string &method_name)
{
    switch (stype)
    {
        case ScalarType::UInt8:
            return sizeOf<ScalarTypeToCppType<ScalarType::UInt8>::type>(1);
        case ScalarType::UInt16:
            return sizeOf<ScalarTypeToCppType<ScalarType::UInt16>::type>(1);
        case ScalarType::UInt32:
            return sizeOf<ScalarTypeToCppType<ScalarType::UInt32>::type>(1);
        case ScalarType::UInt64:
            return sizeOf<ScalarTypeToCppType<ScalarType::UInt64>::type>(1);
        case ScalarType::Int8:
            return sizeOf<ScalarTypeToCppType<ScalarType::Int8>::type>(1);
        case ScalarType::Int16:
            return sizeOf<ScalarTypeToCppType<ScalarType::Int16>::type>(1);
        case ScalarType::Int32:
            return sizeOf<ScalarTypeToCppType<ScalarType::Int32>::type>(1);
        case ScalarType::Int64:
            return sizeOf<ScalarTypeToCppType<ScalarType::Int64>::type>(1);
        
    #ifdef HALF
        case ScalarType::Float16:
            return sizeOf<ScalarTypeToCppType<ScalarType::Float16>::type>(1);
    #endif

        case ScalarType::Float32:
            return sizeOf<ScalarTypeToCppType<ScalarType::Float32>::type>(1);
        case ScalarType::Float64:
            return sizeOf<ScalarTypeToCppType<ScalarType::Float64>::type>(1);
        case ScalarType::Bool:
            return sizeOf<ScalarTypeToCppType<ScalarType::Bool>::type>(1);
    }

    return 0;
}


static inline void* mixScalarAlloc(const size_t size, const ScalarType stype, const DeviceType device_type,
                                   const string &file_name, const string &method_name)
{
    switch (stype)
    {
        case ScalarType::UInt8:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::UInt8>::type>(size, device_type, file_name, method_name);
        case ScalarType::UInt16:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::UInt16>::type>(size, device_type, file_name, method_name);
        case ScalarType::UInt32:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::UInt32>::type>(size, device_type, file_name, method_name);
        case ScalarType::UInt64:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::UInt64>::type>(size, device_type, file_name, method_name);
        case ScalarType::Int8:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Int8>::type>(size, device_type, file_name, method_name);
        case ScalarType::Int16:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Int16>::type>(size, device_type, file_name, method_name);
        case ScalarType::Int32:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Int32>::type>(size, device_type, file_name, method_name);
        case ScalarType::Int64:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Int64>::type>(size, device_type, file_name, method_name);

    #ifdef HALF
        case ScalarType::Float16:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Float16>::type>(size, device_type, file_name, method_name);
    #endif     

        case ScalarType::Float32:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Float32>::type>(size, device_type, file_name, method_name);
        case ScalarType::Float64:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Float64>::type>(size, device_type, file_name, method_name);
        case ScalarType::Bool:
            return mixMemoryAlloc<ScalarTypeToCppType<ScalarType::Bool>::type>(size, device_type, file_name, method_name);
    }

    return nullptr;
}


static inline void mixScalarDeAlloc(void *src, const ScalarType stype, const DeviceType device_type,
                                    const string &file_name, const string &method_name)
{
    switch (stype)
    {
        case ScalarType::UInt8:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::UInt8>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::UInt16:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::UInt16>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::UInt32:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::UInt32>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::UInt64:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::UInt64>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Int8:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Int8>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Int16:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Int16>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Int32:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Int32>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Int64:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Int64>::type*)src, device_type, file_name, method_name);
            break;

    #ifdef HALF
        case ScalarType::Float16:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Float16>::type*)src, device_type, file_name, method_name); 
            break;
    #endif 

        case ScalarType::Float32:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Float32>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Float64:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Float64>::type*)src, device_type, file_name, method_name);
            break;
        case ScalarType::Bool:
            mixMemoryDeAlloc((ScalarTypeToCppType<ScalarType::Bool>::type*)src, device_type, file_name, method_name);
            break;
    }
}

static inline void mixScalarCopy(const void *src, const DeviceType src_device_type,
                                 void *dst, const DeviceType dst_device_type, const ScalarType stype, const size_t size,
                                 const string &file_name, const string &method_name)
{
    switch (stype)
    {
        case ScalarType::UInt8:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt8>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::UInt8>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::UInt16:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt16>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::UInt16>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::UInt32:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt32>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::UInt32>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::UInt64:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::UInt64>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::UInt64>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Int8:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int8>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Int8>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Int16:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int16>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Int16>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Int32:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int32>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Int32>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Int64:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Int64>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Int64>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;

    #ifdef HALF
        case ScalarType::Float16:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float16>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Float16>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
    #endif

        case ScalarType::Float32:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float32>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Float32>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Float64:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Float64>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Float64>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
        case ScalarType::Bool:
            mixMemoryCopy((ScalarTypeToCppType<ScalarType::Bool>::type*)src, src_device_type, 
                          (ScalarTypeToCppType<ScalarType::Bool>::type*)dst, dst_device_type, 
                          size, file_name, method_name);
            break;
    }
}


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp