#pragma once

#include <include/base/InlineProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/tensor/TensorArithmetic.hpp>


namespace ndpp
{

namespace ndpp_arithmetic
{

template<typename T1, typename T2>
ndppInline T1 arithmeticOp(const T1 src1, const T2 src2, const Arithmetic type)
{
    switch (type)
    {
    case Arithmetic::Add:
        return src1 + static_cast<T1>(src2);
    case Arithmetic::Subtract:
        return src1 - static_cast<T1>(src2);
    case Arithmetic::Multiply:
        return src1 * static_cast<T1>(src2);
    case Arithmetic::Division:
        return src1 / static_cast<T1>(src2);
    }

    return static_cast<T1>(0);
}


template<typename T>
ndppInline bool arithmeticOp(const bool src1, const T src2, const Arithmetic type)
{
    switch (type)
    {
    case Arithmetic::Add:
        return src1 || static_cast<bool>(src2);
    case Arithmetic::Subtract:
        return src1 - static_cast<bool>(src2);
    case Arithmetic::Multiply:
        return src1 && static_cast<bool>(src2);
    case Arithmetic::Division:
        return src1 / static_cast<bool>(src2);
    }

    return false;
}


// arithmeticIndexing: Data indexing with multiple dimension for arithmetic.
template<typename T>
ndppInline bool arithmeticIndexing(const T *src, const int64_t index, const int64_t src_size, const int64_t src_dim,
                                   const size_t *acutal_shape, const size_t *allocated_shape, T &dst)
{
    if (src_size == 1)
    {
        dst = src[0];
        return true;
    }

    bool available_access = true;
    int64_t index_tmp = index;
    int64_t actual_index = 0;

    for (int64_t dim_index = src_dim - 1; dim_index >= 0; --dim_index)
    {
        const int64_t each_allocated_size = static_cast<int64_t>(allocated_shape[dim_index]);
        const int64_t each_actual_size = static_cast<int64_t>(acutal_shape[dim_index]);
        const int64_t each_idx = index_tmp % each_allocated_size;
        index_tmp = index_tmp / each_allocated_size;

        // P.S It will be optizimed via allocated strides (Origin: O(n^2), Future: O(n)).
        if (each_idx < each_actual_size)
        {
            int64_t col_size = 1;
            for (int64_t col_index = dim_index + 1; col_index < src_dim; ++col_index)
            {
                col_size *= static_cast<int64_t>(allocated_shape[col_index]);
            }
            
            actual_index += each_idx * col_size;
        }
        else
        {
            available_access = false;
            break;
        }
    }

    if (available_access)
    {
        dst = src[actual_index];
    }

    return available_access;
}

}; // namespace ndpp_arithmetic

}; // namespace ndpp