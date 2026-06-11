#include <omp.h>
#include <iostream>

#include <include/ndpp/ndpp.hpp>

using namespace std;
using namespace ndpp;


#ifdef CUDA
#define device_t 1
#endif
#define scalar_t float


int main()
{

while (1)
{
    // example: 1d-array.
    Array<scalar_t> array_1d(10000, ndpp::Host);
    //Array<SAMPLE_TYPE> array_1d;
    //array_1d.allocate(5, ndpp::Host);

    cout << "array_1d:" << endl;
    cout << "array_1d.size(): " << array_1d.size() << endl;
    cout << "array_1d.device(): " << static_cast<int>(array_1d.device()) << endl;
    cout << "array_1d.deviceStat(): " << static_cast<int>(array_1d.status()) << endl;
    cout << "array_1d.data(): " << array_1d.data() << endl;

    // example: data assignment in 1d-array.
    scalar_t *array_1d_ptr = array_1d.data();
    double start_time_2 = omp_get_wtime();
    for (int index = 0; index < static_cast<int>(array_1d.size()); ++index)
    {
        array_1d_ptr[index] = index + 1;
        //array_1d.at(index) = index + 1;
    }
    double end_time_2 = omp_get_wtime();

    double start_time_1 = omp_get_wtime();
    for (int index = 0; index < static_cast<int>(array_1d.size()); ++index)
    {
        array_1d[index] = index + 1;
        //array_1d.at(index) = index + 1;
    }
    double end_time_1 = omp_get_wtime();


    

     double elapsed_time_1 = end_time_1 - start_time_1;
     double elapsed_time_2 = end_time_2 - start_time_2;

     cout << "elapsed_time_1: " << elapsed_time_1 << endl;
     cout << "elapsed_time_2: " << elapsed_time_2 << endl;


#ifdef CUDA
    ndpp::ndpp_memory::ndpp_cuda::cudaExecDevice(device_t, "array_example.cpp", "main()");
    auto cpy_dtype = ndpp::CudaDevice;
#else
    auto cpy_dtype = ndpp::Host;
#endif
    Array<scalar_t> array_1d_clone = array_1d.clone(cpy_dtype);
    cout << "array_1d_clone:" << endl;
    cout << "array_1d_clone.size(): " << array_1d_clone.size() << endl;
    cout << "array_1d_clone.device(): " << static_cast<int>(array_1d_clone.device()) << endl;
    cout << "array_1d_clone.status(): " << static_cast<int>(array_1d_clone.status()) << endl;
    cout << "array_1d_clone.data(): " << array_1d_clone.data() << endl << endl;

    /*cout << "array_1d_clone: ";
    for (int index = 0; index < static_cast<int>(array_1d_clone.size()); ++index)
    {
        cout << array_1d_clone[index] << ", ";
    }
    cout << endl << endl;*/


    /*cout << "array_1d: ";
    for (int index = 0; index < static_cast<int>(array_1d.size()); ++index)
    {
        cout << array_1d[index] << ", ";
    }
    cout << endl << endl;*/

    // example: data reference in 1d-array.
    Array<scalar_t> &array_1d_ref = array_1d;
    cout << "array_1d_ref:" << endl;
    cout << "array_1d_ref.size(): " << array_1d_ref.size() << endl;
    cout << "array_1d_ref.device(): " << static_cast<int>(array_1d_ref.device()) << endl;
    cout << "array_1d_ref.status(): " << static_cast<int>(array_1d_ref.status()) << endl;
    cout << "array_1d_ref.data(): " << array_1d_ref.data() << endl << endl;
   // cout << &array_1d_ref << ", " << &array_1d << endl;

    // example: data copy in 1d-array.
    Array<scalar_t> array_1d_cpy = array_1d;
    cout << "array_1d_cpy:" << endl;
    cout << "array_1d_cpy.size(): " << array_1d_cpy.size() << endl;
    cout << "array_1d_cpy.device(): " << static_cast<int>(array_1d_cpy.device()) << endl;
    cout << "array_1d_cpy.status(): " << static_cast<int>(array_1d_cpy.status()) << endl;
    cout << "array_1d_cpy.data(): " << array_1d_cpy.data() << endl << endl;
    /*cout << "array_1d_cpy: ";
    for (int index = 0; index < static_cast<int>(array_1d_cpy.size()); ++index)
    {
        cout << array_1d_cpy[index] << ", ";
    }*/
    cout << endl << endl;



    Array<scalar_t> array_1d_list = {1, 2, 3, 4, 5};
    cout << "array_1d_list: ";
    for (int index = 0; index < static_cast<int>(array_1d_list.size()); ++index)
    {
        cout << array_1d_list[index] << ", ";
    }
    cout << endl << endl;
    

    // example: 2d-array.
    Array<Array<scalar_t>> array_2d(2, ndpp::Host);
    for (int index = 0; index < static_cast<int>(array_2d.size()); ++index)
    {
        array_2d[index].allocate(3, ndpp::Host);
    }

    // example: data assignment in 2d-array.
    scalar_t count = 1;
    for (int i = 0; i < static_cast<int>(array_2d.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(array_2d[0].size()); ++j)
        {
            array_2d[i][j] = count;
            ++count;
        }
    }


    cout << "array_2d: " << endl;
    for (int i = 0; i < static_cast<int>(array_2d.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(array_2d[0].size()); ++j)
        {
            cout << &array_2d[i][j] << ", ";
            //cout << array_2d[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl << endl;

    // example: data reference in 2d-array.
    cout << "array_2d_ref: " << endl;
    Array<Array<scalar_t>> &array_2d_ref = array_2d;
    for (int i = 0; i < static_cast<int>(array_2d_ref.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(array_2d_ref[0].size()); ++j)
        {
            cout << &array_2d_ref[i][j] << ", ";
            //cout << array_2d_ref[i][j] << ", ";
        }
        cout << endl;
    }
    cout << endl<< endl;

    // example: data copy in 2d-array.
    cout << "array_2d_cpy: " << endl;
    Array<Array<scalar_t>> array_2d_cpy = array_2d;
    for (int i = 0; i < static_cast<int>(array_2d_cpy.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(array_2d_cpy[0].size()); ++j)
        {
            //cout << &array_2d_cpy[i][j] << ", ";
            cout << array_2d_cpy[i][j] << ", ";
        }
        cout << endl;
    }
}

    return 0;
}