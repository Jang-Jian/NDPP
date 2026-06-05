#include <ctime>
#include <cstdio>

#include <include/logging/Logging.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_log
{

string logger(const string &file_name, const string &method_name, const RuntimeType level, 
              const string &message, const bool shown)
{
    time_t timestamp_now = time(&timestamp_now);
    struct tm *datetime = localtime(&timestamp_now);

    string _now_dt_str = "[" + to_string(datetime->tm_year + 1900) + "-" +
                               to_string(datetime->tm_mon + 1) + "-" +
                               to_string(datetime->tm_mday) + " " +
                               to_string(datetime->tm_hour) + ":" +
                               to_string(datetime->tm_min) + ":" +
                               to_string(datetime->tm_sec) + "]";

    string _log_level_str;
    switch (level)
    {
        case RuntimeType::INFO:  
            _log_level_str = "[RuntimeInfo]";  
            break;
        case RuntimeType::ERROR: 
            _log_level_str = "[RuntimeError]"; 
            break;
        case RuntimeType::WARN:  
            _log_level_str = "[RuntimeWarn]";  
            break;
        case RuntimeType::DEBUG: 
            _log_level_str = "[RuntimeDebug]"; 
            break;
        case RuntimeType::END:   
            _log_level_str = "[RuntimeEnd]";   
            break;
    }

    string _file_str = "[" + file_name + "]";
    string _method_str = "[" + method_name + "]";

    string _system_log = _now_dt_str + _file_str + _method_str + _log_level_str + ": " + message + "\n";
    const char *_system_log_ptr = _system_log.c_str();

    if (shown)
    {
        switch (level)
        {
            case RuntimeType::ERROR:
            case RuntimeType::WARN:
                fprintf(stderr, "%s", _system_log_ptr);
                break;

            case RuntimeType::INFO:
            case RuntimeType::DEBUG:
            case RuntimeType::END:
                fprintf(stdout, "%s", _system_log_ptr);
                break;
        }
    }
    
    return _system_log;
}

}; // namespace ndpp::ndpp_log

}; // namespace ndpp