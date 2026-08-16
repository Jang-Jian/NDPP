#include <include/base/ScalarType.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_memory
{


std::string scalarTypeToCppStr(const ScalarType stype, const bool terminate_out_of_stype)
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

    if (!terminate_out_of_stype)
    {
        ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::scalarTypeToCppStr()", ndpp_log::RuntimeType::Warn, 
                         "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and it will return float32 as default.", true);
    }
    else
    {
        ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::scalarTypeToCppStr()", ndpp_log::RuntimeType::Error, 
                         "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and process is terminated.", true);
        exit(EXIT_FAILURE);
    }

    return std::string(_Float32Str);
}

ScalarType cppStrToScalarType(const std::string stype, const bool terminate_out_of_stype)
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

    if (!terminate_out_of_stype)
    {
        ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::cppStrToScalarType()", ndpp_log::RuntimeType::Warn, 
                         "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and it will return ScalarType::Float32 as default.", true);
    }
    else
    {
        ndpp_log::logger("ScalarType.cpp", "ndpp::ndpp_memory::cppStrToScalarType()", ndpp_log::RuntimeType::Error, 
                         "It doesn't find the correct ndpp::ndpp_memory::ScalarType, and process is terminated.", true);
        exit(EXIT_FAILURE);
    }
    
    return ScalarType::Float32;
}

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp