#include <iostream>

#include <include/base/GenericBase.hpp>

using namespace std;
using namespace ndpp;


static inline int test_int_add_function(const Generic &a, const Generic &b)
{
    return a.value<int>() + b.value<int>();
}

int main()
{
    while (1)
    {
    Generic a(5);
    Generic b;
    b.set(string("abc"));

    cout << "a: " << a.value<int>() << endl;
    cout << "b: " << b.value<string>() << endl;
    cout << "add: " << test_int_add_function(5, 10) << endl;
    }

    return 0;
}