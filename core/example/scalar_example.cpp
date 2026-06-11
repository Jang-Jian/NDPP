#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;

int main()
{
    cout << "sizeof(scalar_t): " << sizeof(ndpp_memory::ScalarTypeToCppType<ndpp::Float32>::type) << endl;

    auto stype = ndpp_memory::CppTypeToScalarType<int>::value;
    cout << "stype: " << int(stype) << endl;

    return 0;
}