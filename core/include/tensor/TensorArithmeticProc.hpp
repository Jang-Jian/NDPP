#pragma once

#include <include/base/InlineProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/base/SizeProc.hpp>
#include <include/base/BasicType.hpp>
#include <include/tensor/TensorArithmetic.hpp>


namespace ndpp
{

namespace ndpp_arithmetic
{

/*
    Arithmetic Operator (V2) using template specialization.
*/
#define NDPP_DEFINE_BINARY_OP(OperatorName, Basic_Expr, Bool_Expr)    \
struct OperatorName                                                   \
{                                                                     \
    template<typename T1, typename T2>                                \
    ndppInline T1 operator()(T1 lhs, T2 rhs) const                    \
    {                                                                 \
        const T1 r = static_cast<T1>(rhs);                            \
        return (Basic_Expr);                                          \
    }                                                                 \
                                                                      \
    template<typename T>                                              \
    ndppInline bool operator()(bool lhs, T rhs) const                 \
    {                                                                 \
        const bool r = static_cast<bool>(rhs);                        \
        return (Bool_Expr);                                           \
    }                                                                 \
};

NDPP_DEFINE_BINARY_OP(AddOp, lhs + r, lhs || r) 
NDPP_DEFINE_BINARY_OP(SubOp, lhs - r, lhs - r)
NDPP_DEFINE_BINARY_OP(MulOp, lhs * r, lhs && r)
NDPP_DEFINE_BINARY_OP(DivOp, lhs / r, lhs / r)


// arithmeticOp(V1).
template<typename T1, typename T2>
ndppInline T1 arithmeticOp(const T1 lhs, const T2 rhs, const Arithmetic type)
{
    const T1 r = static_cast<T1>(rhs);

    switch (type)
    {
    case Arithmetic::Add:
        return lhs + r;
    case Arithmetic::Subtract:
        return lhs - r;
    case Arithmetic::Multiply:
        return lhs * r;
    case Arithmetic::Division:
        return lhs / r;
    }

    return static_cast<T1>(0);
}


// arithmeticOp(V1).
template<typename T>
ndppInline bool arithmeticOp(const bool lhs, const T rhs, const Arithmetic type)
{
    const bool r = static_cast<bool>(rhs);

    switch (type)
    {
    case Arithmetic::Add:
        return lhs || r;
    case Arithmetic::Subtract:
        return lhs - r;
    case Arithmetic::Multiply:
        return lhs && r;
    case Arithmetic::Division:
        return lhs / r;
    }

    return false;
}



// arithmeticIndexing(V1): Data indexing with multiple dimension for arithmetic.
template<typename T>
ndppInline bool arithmeticIndexing(const T *src, const int64_t index, const int64_t src_size, const int64_t src_dim,
                                   const Integer *acutal_shape, const Integer *allocated_shape, T &dst)
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


// arithmeticIndexingV2(V2): Data indexing with multiple dimension for arithmetic.
template<typename T>
ndppInline bool arithmeticIndexingV2(const T* __restrict__ src, 
                                     int64_t index, 
                                     int64_t src_size,
                                     int64_t src_dim,
                                     const Integer* __restrict__ actual_shape, 
                                     const Integer* __restrict__ allocated_shape, 
                                     const Integer* __restrict__ strides, 
                                     T &dst)
{
    if (src_size == 1)
    {
        dst = src[0];
        return true;
    }

    int64_t actual_index = 0;
    constexpr size_t scalar_type_size = sizeof(T); // size of data type (sizeof(T)).

    for (int64_t d = src_dim - 1; d >= 0; --d)
    {
        int64_t idx = index % allocated_shape[d];
        index /= allocated_shape[d];

        if (idx >= static_cast<int64_t>(actual_shape[d]))
            return false;

        actual_index += idx * (strides[d] / scalar_type_size);
    }

    dst = src[actual_index];
    return true;
}


}; // namespace ndpp_arithmetic

}; // namespace ndpp