#pragma once

#include <iostream>

using namespace std;


namespace ndpp
{

namespace ndpp_log
{ 

constexpr const char* _InfoStr  = "info";
constexpr const char* _ErrorStr = "error";
constexpr const char* _WarnStr  = "warn";
constexpr const char* _DebugStr = "debug";
constexpr const char* _EndStr   = "end";


// LogType: log type.
enum class RuntimeType
{
    // [RuntimeInfo].
    Info  = 0,

    // [RuntimeError].
    Error = 1,

    // [RuntimeWarn].
    Warn  = 2,

    // [RuntimeDebug].
    Debug = 3,

    // [RuntimeEnd].
    End   = 4
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