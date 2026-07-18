#pragma once

#include <include/base/ScalarType.hpp>
#include <include/tensor/TensorArithmetic.hpp>
#include <include/tensor/TensorArithmeticProc.hpp>


namespace ndpp
{

namespace ndpp_arithmetic
{

// cpuArithForwardACTypeImpleKernel: Deciding the data type for a & c.
// P.S Wrapping for gpuArithForwardBTypeImpleKernel() with different data type a & c.
void gpuArithForwardACTypeImpleKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                                      const size_t *a_allocated_shape, const size_t *a_acutal_shape, const int64_t a_shape_dim, const int64_t a_total,
                                      const size_t *b_allocated_shape, const size_t *b_acutal_shape, const int64_t b_shape_dim, const int64_t b_total,
                                      const int64_t loop_total, const Arithmetic arith_type);

}; // namespace ndpp_arithmetic

}; // namespace ndpp