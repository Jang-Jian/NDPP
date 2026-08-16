#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;


int main()
{
//while (1)
//{
    Stack test_stack;

    test_stack.push(6);
    test_stack.push(std::string("abc"));
    test_stack.push(5.0f);
    test_stack.push(ndpp::zeros({3, 4, 5}));

    Stack test_stack2 = test_stack.clone();
    cout << "test_stack's size: " << test_stack.size() << endl;

    for (; !test_stack.empty(); test_stack.pop())
    {
        cout << test_stack.top().name() << endl;
    }

    cout << "============" << endl;

    
    cout << "test_stack2's size: " << test_stack2.size() << endl;
    for (; !test_stack2.empty(); test_stack2.pop())
    {
        cout << test_stack2.top().name() << endl;
    }
//}

    return 0;
}