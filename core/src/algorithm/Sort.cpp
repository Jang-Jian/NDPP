#include <include/algorithm/Sort.hpp>
#include <include/logging/Logging.hpp>
#include <include/tensor/TensorHelper.hpp>

using namespace std;

namespace ndpp
{


// qsortPivot: Comparing with pivot, and selecting the new pivot (used for Quick sort).
template<typename T>
static inline int64_t qsortPivot(T *dst, const int64_t index_strides, const int64_t start_index, 
                                 const int64_t end_index, const bool descending)
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


// qsort: Quick sort.
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


// merge: Comparing left & right segment (used for merge sort).
template<typename T>
static inline void merge(T *dst, const int64_t index_strides, const int64_t start_index, const int64_t medium_index, 
                         const int64_t end_index, const bool descending)
{
    Array<T> left_part(medium_index - start_index + 1, ndpp_memory::DeviceType::Host);
    Array<T> right_part(end_index - medium_index, ndpp_memory::DeviceType::Host);


    const auto dist = [&](Array<T> &part, const int64_t &t1, const int64_t &t2)
    {
        for (int64_t idx = t1; idx <= t2; ++idx)
        {
            part[(idx - t1) * index_strides] = dst[idx * index_strides];
        }
    };

    const auto concat = [&](const Array<T> &part, const int64_t &t1, const int64_t &t2, int64_t &dst_index)
    {
        for (int64_t index = t1; index < t2; ++index)
        {
            dst[dst_index * index_strides] = left_part[index * index_strides];
            ++dst_index;
        }
    };


    dist(left_part, start_index, medium_index);
    dist(right_part, medium_index + 1, end_index);
    

    int64_t dst_i = start_index, left_i = 0, right_i = 0;
    while (left_i < static_cast<int64_t>(left_part.size()) && right_i < static_cast<int64_t>(right_part.size()))
    {
        switch (descending)
        {
            case true:
                if (left_part[left_i * index_strides] > right_part[right_i * index_strides])
                {
                    dst[dst_i * index_strides] = left_part[left_i * index_strides];
                    ++left_i;
                }
                else
                {
                    dst[dst_i * index_strides] = right_part[right_i * index_strides];
                    ++right_i;
                }
                break;
            case false:
                if (left_part[left_i * index_strides] < right_part[right_i * index_strides])
                {
                    dst[dst_i * index_strides] = left_part[left_i * index_strides];
                    ++left_i;
                }
                else
                {
                    dst[dst_i * index_strides] = right_part[right_i * index_strides];
                    ++right_i;
                }
                break;
        }

        ++dst_i;
    }

    // copy the remaining left_part & right_part to dst.
    concat(left_part, left_i, static_cast<int64_t>(left_part.size()), dst_i);
    concat(right_part, right_i, static_cast<int64_t>(right_part.size()), dst_i);
}


// msort: Merge sort.
template<typename T>
static void msort(T *dst, const int64_t index_strides, const int64_t start_index, const int64_t end_index, const bool descending)
{
    if (start_index < end_index)
    {
        const int64_t medium_index = start_index + static_cast<int64_t>((end_index - start_index) / 2);

        msort(dst, index_strides, start_index, medium_index, descending);
        msort(dst, index_strides, medium_index + 1, end_index, descending);

        merge(dst, index_strides, start_index, medium_index, end_index, descending);
    }
}


template <typename T>
using SortAlgo = void (*)(T*, const int64_t, const int64_t, const int64_t, const bool);

// conductSort: Conducting the sorting algorithm.
template<typename T>
static inline void conductSort(T *_dst_ptr, const int64_t group_size, const int64_t index_strides, 
                               const int64_t sorted_size, const bool descending, SortAlgo<T> _sort)
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


// getSortAlgo: Getting the sorting algorithm via stable flag.
template<typename T>
static inline SortAlgo<T> getSortAlgo(const bool stable)
{
    if (stable)
    {
        return &msort<T>;
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
    IntArray _shape = _dst.sizes();
    
    if (dim < 0)
    {
        dim += _self_dim;
    }

    if (dim < 0 || dim >= _self_dim)
    {
        ndpp_log::logger("sort.cpp", "ndpp::sort()", ndpp_log::RuntimeType::Error, 
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
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt16: 
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt32: 
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>(stable));
            break;
        case ndpp_memory::ScalarType::UInt64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int8:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int16:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int32:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Int64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(stable));
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>(stable));
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>(stable));

            break;
        case ndpp_memory::ScalarType::Float64:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(stable));
            break;
        case ndpp_memory::ScalarType::Bool:
            conductSort(static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(_dst.data()), 
                        group_size, index_strides, sorted_size, descending, 
                        getSortAlgo<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(stable));
            break;
    }

    if (src.device() == ndpp_memory::DeviceType::Host)
    {
        return _dst;
    }

    return _dst.clone(src.device()); 
}

};