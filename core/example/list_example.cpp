#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;


void push(List &test)
{
    test.pushFront(6);
    test.pushBack(std::string("abc"));
    test.pushFront(5.0f);
    test.pushFront(ndpp::zeros({3, 4, 5}));
}

int main()
{
    //while (1)
    //{
    List test;
    
    push(test);

    cout << "size: " << test.size() << endl;
    for (NodeDevice node : test)
    {
        cout << node.data().name() << endl;
    }
    
    cout << "=========" << endl;
    test.insert(ndpp::next(test.begin(), 3), 'a', ndpp::Front);
     //test.insert(ndpp::next(test.begin(), 3), 'b', ndpp::Front);

    cout << "size: " << test.size() << endl;
    for (auto node : test)
    {
        cout << node.data().name() << endl;
    }


    ndpp::ListIterator it = ndpp::next(test.begin(), 0);

    NodeDevice node = *it;
    Tensor &tensor_test = node.data().value<ndpp::Tensor>();


    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << float() << endl;
                tensor_test[i][j][k] = float(k + j * (5)+ i * 20);
            }
        }
    }

    cout << "=========" << endl;
    test.erase(ndpp::next(test.begin(), 3));
    //test.popFront();
    //test.popBack();

    cout << "size: " << test.size() << endl;
    for (auto node : test)
    {
        cout << node.data().name() << endl;
    }

    // Confirm the data whether they are inserted.
    Tensor &tensor_test2 = node.data().value<ndpp::Tensor>();
    cout << "tensor_test:" << endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(tensor_test2[i][j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    // Retrive the previous node from tail (P.S we use --it to do reverse loop).
    cout << "=========" << endl;
    for (auto it = test.rbegin(); it != test.rend(); --it) 
    {
        cout << (*it).data().name() << endl;
    }

    // Example. splice.
    List test2 = test.clone();
    test.splice(ndpp::next(test.begin(), 3), test2);
    cout << "=========" << endl;
    cout << "size: " << test.size() << endl;
    for (auto node : test)
    {
        cout << node.data().name() << endl;
    }
    //}

    return 0;
}