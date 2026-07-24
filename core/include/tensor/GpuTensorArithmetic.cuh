#pragma once

#include <include/base/ScalarType.hpp>
#include <include/tensor/TensorArithmetic.hpp>
#include <include/tensor/TensorArithmeticProc.hpp>


namespace ndpp
{

namespace ndpp_arithmetic
{

void gpuArithForwardKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                           const size_t *a_allocated_shape, const size_t *a_acutal_shape, const size_t *a_strides, int64_t a_shape_dim, int64_t a_total,
                           const size_t *b_allocated_shape, const size_t *b_acutal_shape, const size_t *b_strides, int64_t b_shape_dim, int64_t b_total,
                           int64_t loop_total, Arithmetic arith_type);

}; // namespace ndpp_arithmetic

}; // namespace ndpp