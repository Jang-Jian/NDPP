#include <include/base/ScalarType.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_memory
{

static constexpr const char* _UInt8Str   = "uint8";
static constexpr const char* _UInt16Str  = "uint16";
static constexpr const char* _UInt32Str  = "uint32";
static constexpr const char* _UInt64Str  = "uint64";
static constexpr const char* _Int8Str    = "int8";
static constexpr const char* _Int16Str   = "int16";
static constexpr const char* _Int32Str   = "int32";
static constexpr const char* _Int64Str   = "int64";
#ifdef HALF
static constexpr const char* _Float16Str = "float16";
#endif
static constexpr const char* _Float32Str = "float32";
static constexpr const char* _Float64Str = "float64";
static constexpr const char* _BoolStr    = "bool";


std::string scalarTypeToCppStr(const ScalarType stype)
{
    switch (stype)
    {
        case ScalarType::UInt8:  
            return std::string(_UInt8Str);
        case ScalarType::UInt16: 
            return std::string(_UInt16Str);
        case ScalarType::UInt32: 
            return std::string(_UInt32Str);
        case ScalarType::UInt64:
            return std::string(_UInt64Str);
        case ScalarType::Int8:
            return std::string(_Int8Str);
        case ScalarType::Int16:
            return std::string(_Int16Str);
        case ScalarType::Int32:
            return std::string(_Int32Str);
        case ScalarType::Int64:
            return std::string(_Int64Str);
    #ifdef HALF
        case ScalarType::Float16:
            return std::string(_Float16Str);
    #endif
        case ScalarType::Float32:
            return std::string(_Float32Str);
        case ScalarType::Float64:
            return std::string(_Float64Str);
        case ScalarType::Bool:
            return std::string(_BoolStr);
    }

    ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::scalarTypeToCppStr()", ndpp_log::RuntimeType::WARN, 
                     "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and it will return float32 as default.", true);

    return std::string(_Float32Str);
}

ScalarType cppStrToScalarType(const std::string stype)
{
    if      (!stype.compare(_UInt8Str))   { return ScalarType::UInt8;   }
    else if (!stype.compare(_UInt16Str))  { return ScalarType::UInt16;  }
    else if (!stype.compare(_UInt32Str))  { return ScalarType::UInt32;  }
    else if (!stype.compare(_UInt64Str))  { return ScalarType::UInt64;  }
    else if (!stype.compare(_Int8Str))    { return ScalarType::Int8;    }
    else if (!stype.compare(_Int16Str))   { return ScalarType::Int16;   }
    else if (!stype.compare(_Int32Str))   { return ScalarType::Int32;   }
    else if (!stype.compare(_Int64Str))   { return ScalarType::Int64;   }
#ifdef HALF
    else if (!stype.compare(_Float16Str)) { return ScalarType::Float16; }
#endif
    else if (!stype.compare(_Float32Str)) { return ScalarType::Float32; }
    else if (!stype.compare(_Float64Str)) { return ScalarType::Float64; }
    else if (!stype.compare(_BoolStr))    { return ScalarType::Bool;    }

    ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::cppStrToScalarType()", ndpp_log::RuntimeType::WARN, 
                     "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and it will return ScalarType::Float32 as default.", true);
    
    return ScalarType::Float32;
}

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp