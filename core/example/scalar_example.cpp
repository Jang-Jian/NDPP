#include <iostream>

#include <include/ndpp.hpp>

using namespace std;
using namespace ndpp;

int main()
{
    cout << "sizeof(scalar_t): " << sizeof(ndpp_memory::ScalarTypeToCppType<ndpp::ndpp_memory::ScalarType::Float32>::type) << endl;

    ndpp::ndpp_memory::ScalarType stype = ndpp_memory::CppTypeToScalarType<int>::value;
    cout << "stype: " << int(stype) << endl;

    return 0;
}