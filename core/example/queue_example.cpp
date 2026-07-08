#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;


int main()
{
//while (1)
//{
    Queue test_queue;

    test_queue.push(6);
    test_queue.push(std::string("abc"));
    test_queue.push(5.0f);
    test_queue.push(ndpp::zeros({3, 4, 5}));

    Queue test_queue2 = test_queue.clone();
    cout << "test_queue's size: " << test_queue.size() << endl;

    for (; !test_queue.empty(); test_queue.pop())
    {
        cout << test_queue.front().name() << endl;
    }

    cout << "============" << endl;

    
    cout << "test_queue2's size: " << test_queue2.size() << endl;
    for (; !test_queue2.empty(); test_queue2.pop())
    {
        cout << test_queue2.front().name() << endl;
    }
//}

    return 0;
}