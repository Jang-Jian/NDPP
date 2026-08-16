#pragma once

#include <omp.h>

#include <include/base/InlineProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/base/Parallelism.hpp>


namespace ndpp
{

namespace ndpp_memory
{


// Accessing the value from void*.
template<typename T>
ndppInline T cvtValue(const void *value, const ScalarType value_stype);


// Specifying the value at specific memory address.
void memset(void *dst, const void *value, 
            const ScalarType dst_stype, const ScalarType value_stype,
            const DeviceType dst_dtype, const size_t size, 
            const string &file_name, const string &method_name);


// Deep copy via different data type.
void cvtScalarType(const void *src, const ScalarType src_stype, 
                   void *dst, const ScalarType dst_stype, const DeviceType dtype, 
                   const size_t size);


                   
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

    return static_cast<T>(static_cast<double>(0));
}


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp