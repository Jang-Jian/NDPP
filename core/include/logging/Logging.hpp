#pragma once

#include <iostream>

using namespace std;


namespace ndpp
{

namespace ndpp_log
{

// LogType: log type.
enum class RuntimeType
{
    // [RuntimeInfo].
    INFO  = 0,

    // [RuntimeError].
    ERROR = 1,

    // [RuntimeWarn].
    WARN  = 2,

    // [RuntimeDebug].
    DEBUG = 3,

    // [RuntimeEnd].
    END   = 4
};


/*
    Method Logging.
    Example.
    - logger("test.cpp", "test()", RuntimeType::INFO, "testing", true);
    - [2026-6-6 6:6:6][test.cpp][logging()][RuntimeInfo]: testing.
*/
string logger(const string &file_name, const string &method_name, const RuntimeType level, 
              const string &message, const bool shown);

}; // namespace ndpp::ndpp_log

}; // namespace ndpp