#include <omp.h>
#include <iostream>

#include <include/ndpp.hpp>

using namespace std;
using namespace ndpp;

#ifdef CUDA
#define device_t 1
#endif
#define scalar_t ndpp_memory::ScalarType::Float32


int main()
{

//while (1)
//    {
#ifdef CUDA
    ndpp::ndpp_memory::ndpp_cuda::cudaExecDevice(device_t, "tensor_example.cpp", "main()");
    auto dtype = ndpp_memory::DeviceType::CudaDevice;
#else
    auto dtype = ndpp_memory::DeviceType::Host;
#endif
    Tensor tensor_test;
    tensor_test.zerosB({3, 4, 5}, scalar_t, dtype);

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


    

    double start_time_1 = omp_get_wtime();
    //#pragma omp parallel for
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
    double end_time_1 = omp_get_wtime();
    cout << endl;


#ifdef CUDA
    Tensor tensor_test2;
    tensor_test2.copy(tensor_test, ndpp_memory::DeviceType::CudaDevice);

    cout << "tensor_test2:" << endl;
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
#endif

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

    
    /*double start_time_2 = omp_get_wtime();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                //cout << float() << endl;
                int index = k + j * (5)+ i * 20;
                tensor_ptr[index] = float(index);
            }
        }
    }
    double end_time_2 = omp_get_wtime();
    cout << endl;*/

    double elapsed_time_1 = end_time_1 - start_time_1;
    //double elapsed_time_2 = end_time_2 - start_time_2;

     cout << "elapsed_time_1: " << elapsed_time_1 << endl;
     //cout << "elapsed_time_2: " << elapsed_time_2 << endl;
//}


    return 0;
}