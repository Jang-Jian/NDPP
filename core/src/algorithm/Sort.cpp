#include <include/algorithm/Sort.hpp>
#include <include/logging/Logging.hpp>
#include <include/tensor/TensorHelper.hpp>

using namespace std;

namespace ndpp
{



template<typename T>
static inline int64_t qsortPivot(T *dst, const int64_t index_strides, 
                                const int64_t start_index, const int64_t end_index, const bool descending)
{
    T pivot_data = dst[end_index * index_strides];
    int64_t move_index = start_index;

    for (int64_t index = start_index; index < end_index; ++index)
    {
        switch (descending)
        {
            case true:
                if (dst[index * index_strides] > pivot_data)
                {
                    std::swap(dst[index * index_strides], dst[move_index * index_strides]);
                    ++move_index;
                }
                break;
            
            case false:
                if (dst[index * index_strides] < pivot_data)
                {
                    std::swap(dst[index * index_strides], dst[move_index * index_strides]);
                    ++move_index;
                }
                break;
        }
    }

    std::swap(dst[end_index * index_strides], dst[move_index * index_strides]);
    return move_index;
}

template<typename T>
static void qsort(T *dst, const int64_t index_strides, const int64_t start_index, const int64_t end_index, const bool descending)
{
    if (start_index < end_index)
    {
        const int64_t pivot = qsortPivot(dst, index_strides, start_index, end_index, descending);

        qsort(dst, index_strides, start_index, pivot - 1, descending);
        qsort(dst, index_strides, pivot + 1, end_index, descending);
    }
}


// Conducting the sort algorithm.
template<typename T>
static inline void conductSort(T *_dst_ptr, const int64_t group_size, const int64_t index_strides, 
                               const int64_t sorted_size, const bool descending, 
                               void (*_sort)(T*, const int64_t, const int64_t, const int64_t, const bool))
{
    for (int64_t row = 0; row < group_size; ++row)
    {
        for (int64_t col = 0; col < index_strides; ++col)
        {
            _sort(_dst_ptr + (col + row * (index_strides * sorted_size)), 
                  index_strides, 0, sorted_size - 1, descending);
        }
    }
}


template <typename T>
using SortMethod = void (*)(T*, const int64_t, const int64_t, const int64_t, const bool);

template<typename T>
static inline SortMethod<T> getSort(const bool stable)
{
    // TODO: It will be chagned to merge sort.
    if (stable)
    {
        return &qsort<T>;
    }

    return &qsort<T>;
}


Tensor sort(const Tensor &src, int64_t dim, const bool descending, const bool stable)    
{
    if (src.dim() == 0)
    {
        return Tensor();
    }

    Tensor _dst = src.clone(ndpp_memory::DeviceType::Host);

    int64_t _self_dim = static_cast<int64_t>(_dst.dim());
    SizeTArray _shape = _dst.sizes();
    
    if (dim < 0)
    {
        dim += _self_dim;
    }

    if (dim < 0 || dim >= _self_dim)
    {
        ndpp_log::logger("sort.cpp", "ndpp::sort()", ndpp_log::RuntimeType::ERROR, 
                         "Dim " + to_string(static_cast<long long>(dim)) + " is out of dimension with size " + 
                         to_string(static_cast<long long>(_self_dim)) + ".", true);
        exit(EXIT_FAILURE);
    }

    
    int64_t group_size = 1;
    int64_t sorted_size = static_cast<int64_t>(_shape[static_cast<size_t>(dim)]);
    int64_t index_strides = 1;

    // group_size: Calculating the how many list will be sorted.
    for (int64_t index = 0; index < dim; ++index)
    {
        group_size *= _shape[index];
    }

    // index_strides: Calculating the stride size for indexing.
    for (int64_t index = dim + 1; index < _self_dim; ++index)
    {
        index_strides *= _shape[index];
    }

    /*cout << "group_size: " << group_size << endl;
    cout << "index_strides: " << index_strides << endl;
    cout << "sorted_size: " << sorted_size << endl;
    cout << "dim: " << dim << endl;*/

    // Sorting along the specific dim.
    switch (_dst.scalar())
    {
        case ndpp_memory::ScalarType::UInt8:  
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt16: 
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt32: 
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int8:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int16:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int32:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(stable));
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>(stable));
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>(stable));

            break;
        case ndpp_memory::ScalarType::Float64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Bool:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSort<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(stable));
            break;
    }

    return _dst.clone(src.device()); 
}

};