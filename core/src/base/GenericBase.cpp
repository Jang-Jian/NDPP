#include <cxxabi.h>
#include <memory>

#include <include/base/GenericBase.hpp>


namespace ndpp
{

const std::string Generic::typeInfo2Str(const std::type_info& info) const
{
    int status = 0;
    // __cxa_demangle allocates memory via malloc; using a unique_ptr handles free()
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(info.name(), nullptr, nullptr, &status),
        std::free
    );
    return (status == 0) ? std::string(demangled.get()) : std::string(info.name());
}

}; // namespace ndpp