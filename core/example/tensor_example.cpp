#include <omp.h>
#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;

#ifdef CUDA
#define device_id_t 1
#define device_t ndpp::CudaZeroCpy
#else
#define device_t ndpp::Host
#endif

#define scalar_t ndpp::Float32


int main()
{

//while (1)
 //   {
#ifdef CUDA
    ndpp::ndpp_memory::ndpp_cuda::cudaExecDevice(device_id_t, "tensor_example.cpp", "main()");
#endif
    Tensor tensor_test = ndpp::zeros({3, 4, 5}, scalar_t, device_t);
    Tensor tensor_test2 = ndpp::zeros({4, 5}, scalar_t, device_t);

    cout << "tensor_test ptr: " << tensor_test.data() << endl;
    cout << "tensor_test dim: " << tensor_test.dim() << endl;
    cout << "tensor_test stype: " << int(tensor_test.scalar()) << endl;

    SizeTArray shape = tensor_test.sizes();
    SizeTArray strides = tensor_test.strides();

    cout << "tensor_test shape: ";
    for (auto sh : shape)
    {
        cout << sh << "x";
    }
    cout << endl;

    cout << "tensor_test strides: ";
    for (auto st : strides)
    {
        cout << st << "x";
    }
    cout << endl;


    

    //double start_time_1 = omp_get_wtime();
    for (int i = 0; i < (int)shape[0]; ++i)
    {
        for (int j = 0; j < (int)shape[1]; ++j)
        {
            for (int k = 0; k < (int)shape[2]; ++k)
            {
                //cout << float() << endl;
                 tensor_test2[j][k] = k + j * 5;
                tensor_test[i][j][k] = float(k + j * (5)+ i * 20);
            }
        }
    }
    //double end_time_1 = omp_get_wtime();
    //cout << endl;


    for (int j = 0; j < (int)shape[1]; ++j)
        {
            for (int k = 0; k < (int)shape[2]; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(tensor_test2[j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;


    //float *tensor_ptr = (ndpp_memory::ScalarTypeToCppType<scalar_t>::type*)tensor_test.data();

    cout << "tensor_test:" << endl;
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

    Tensor test2 = tensor_test + tensor_test2 ;

    Tensor tensor_test3 = 10 * tensor_test; 




     cout << "test2:" << endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(test2[i][j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;


    cout << "test3:" << int(tensor_test.scalar()) << endl;
    cout << "test3's device: " << int(tensor_test3.device()) << endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << tensor_ptr[k + j * (5)+ i * 20] << ",";
                cout << float(tensor_test3[i][j][k])  << ",";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    return 0;
}