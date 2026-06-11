#include <omp.h>
#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;

int main()
{
    Tensor tensor_test = ndpp::zeros({3, 4, 5}, ndpp::Float32, ndpp::Host);


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

    double start_time = omp_get_wtime();
    Tensor tensor_sort = ndpp::sort(tensor_test, 1, true);
    double end_time = omp_get_wtime();

    cout << "Origin: " << endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(tensor_test[i][j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    cout << "Sort: " << endl;;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(tensor_sort[i][j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    cout << "Time for sorting: " << end_time - start_time << endl;

    return 0;
}