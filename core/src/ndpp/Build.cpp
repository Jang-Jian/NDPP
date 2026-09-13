#include <include/ndpp/Build.hpp>


namespace ndpp
{

std::string buildInfo()
{
    std::string _build_tags;

    // Build tags.
#ifdef PYTHON
    _build_tags += "WITH_PYTHON=1";
#else
    _build_tags += "WITH_PYTHON=0";
#endif
#ifdef CUDA 
    _build_tags += " WITH_CUDA=1";
#else
    _build_tags += " WITH_CUDA=0";
#endif
#ifdef HALF 
    _build_tags += " WITH_HALF=1";
#else
    _build_tags += " WITH_HALF=0";
#endif

    std::string _os_tags;

    // OS.
#if defined(_WIN32)
    _os_tags = "Win32 (32bits)";
#elif defined(_WIN64)
    _os_tags = "Win32 (64its)";
/*#elif defined(__APPLE__) || defined(__MACH__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        std::cout << "iOS\n";
    #elif TARGET_OS_MAC
        std::cout << "macOS\n";
    #endif*/
#elif defined(__linux__)
    _os_tags = "Linux";
#elif defined(__unix__) || defined(__unix)
    _os_tags = "Unix";
#else
    _os_tags = "Unknown OS";
#endif

    std::string _dst = "The build tags (OS: " + _os_tags + "): " + _build_tags + ".";
    
    return _dst;
}

};