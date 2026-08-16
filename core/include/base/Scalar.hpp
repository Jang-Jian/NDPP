#pragma once

#include <include/base/ScalarType.hpp>
#ifdef HALF
#include <include/half/half.hpp>
#endif


namespace ndpp
{

    
/*
    Scalar: Representing a 0-dimensional tensor which contains a single element.
    Reference: https://github.com/pytorch/pytorch/blob/main/c10/core/Scalar.h#L36
*/
class Scalar 
{
private:
    union ScalarVal
    {
        constexpr ScalarVal() : float64_val(0.0) {}     // user-provided constructor
        //constexpr ~ScalarVal() {}    // user-provided destructor

        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type   uint8_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type  uint16_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type  uint32_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type  uint64_val;

        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type    int8_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type   int16_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type   int32_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type   int64_val;

    #ifdef HALF
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type float16_val;
    #endif
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type float32_val;
        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type float64_val;

        ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type    bool_val;
        //std::complex<double> z;
    };
    
    ScalarVal _data;
    ndpp_memory::ScalarType _stype;

public:
    inline Scalar();
    virtual inline ~Scalar() = default;
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type src);
#ifdef HALF
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type src);
#endif
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src);
    inline Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type src);

    //Scalar(std::complex<double>);

    // Get value via void*.
    inline void* data() const;

    // Get the ndpp::ndpp_memory::ScalarType.
    inline ndpp_memory::ScalarType type() const;
};

inline Scalar::Scalar() : Scalar(ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type(0.0))
{
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type src) 
{ 
    this->_data.uint8_val = src; 
    this->_stype = ndpp_memory::ScalarType::UInt8; 
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type src) 
{ 
    this->_data.uint16_val = src; 
    this->_stype = ndpp_memory::ScalarType::UInt16; 
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type src) 
{
    this->_data.uint32_val = src; 
    this->_stype = ndpp_memory::ScalarType::UInt32;
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type src) 
{ 
    this->_data.uint64_val = src;
    this->_stype = ndpp_memory::ScalarType::UInt64; 
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type src) 
{
    this->_data.int8_val = src; 
    this->_stype = ndpp_memory::ScalarType::Int8; 
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type src) 
{ 
    this->_data.int16_val = src; 
    this->_stype = ndpp_memory::ScalarType::Int16; 
}
    
inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type src) 
{ 
    this->_data.int32_val = src; 
    this->_stype = ndpp_memory::ScalarType::Int32; 
}
    
inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type src) 
{ 
    this->_data.int64_val = src; 
    this->_stype = ndpp_memory::ScalarType::Int64; 
}

#ifdef HALF
inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type src) 
{ 
    this->_data.float16_val = src; 
    this->_stype = ndpp_memory::ScalarType::Float16; 
}
#endif

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type src) 
{ 
    this->_data.float32_val = src; 
    this->_stype = ndpp_memory::ScalarType::Float32; 
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src) 
{ 
    this->_data.float64_val = src; 
    this->_stype = ndpp_memory::ScalarType::Float64;
}

inline Scalar::Scalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type src) 
{ 
    this->_data.bool_val = src; 
    this->_stype = ndpp_memory::ScalarType::Bool; 
}

inline void* Scalar::data() const
{
    // Lazy for using template.
    switch (this->_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            return (void*)&_data.uint8_val;
        case ndpp_memory::ScalarType::UInt16: 
            return (void*)&_data.uint16_val;
        case ndpp_memory::ScalarType::UInt32: 
            return (void*)&_data.uint32_val;
        case ndpp_memory::ScalarType::UInt64:
            return (void*)&_data.uint64_val;
        case ndpp_memory::ScalarType::Int8:
            return (void*)&_data.int8_val;
        case ndpp_memory::ScalarType::Int16:
            return (void*)&_data.int16_val;
        case ndpp_memory::ScalarType::Int32:
            return (void*)&_data.int32_val;
        case ndpp_memory::ScalarType::Int64:
            return (void*)&_data.int64_val;
    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            return (void*)&_data.float16_val;
    #endif
        case ndpp_memory::ScalarType::Float32:
            return (void*)&_data.float32_val;
        case ndpp_memory::ScalarType::Float64:
            return (void*)&_data.float64_val;
        case ndpp_memory::ScalarType::Bool:
            return (void*)&_data.bool_val;

        default:
            break; 
    }

    ndpp_log::logger("ndpp_memory::ScalarType.hpp", "Scalar::data()", ndpp_log::RuntimeType::Warn, 
                     "It doesn't find the correct ndpp::ndpp_memory::ndpp_memory::ScalarType, and it will return nullptr.", true);

    return nullptr;
}

inline ndpp_memory::ScalarType Scalar::type() const
{
    return this->_stype;
}

};