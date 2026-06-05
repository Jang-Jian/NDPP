#pragma once

#include <iostream>

#include <include/base/ScalarType.hpp>
#include <include/array/ArrayStype.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_data_arch
{


// Get the correct shape via 'src_shape' & 'src_strides'.
void calcShape(const SizeTArray &src_shape, const SizeTArray &src_strides, 
               SizeTArray &dst_shape, const ndpp_memory::DeviceType dst_shape_dtype,
               const string &file_name, const string &method_name);


// Get the correct strides via 'src_shape' & data type(dtype).
// p.s. Please reference how numpy.ndarray.strides works.
//      https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.strides.html
void calcStrides(const SizeTArray &src_shape, const ndpp_memory::ScalarType stype,
                 SizeTArray &dst_strides, const ndpp_memory::DeviceType dst_strides_dtype,
                 const string &file_name, const string &method_name);


// Get the size which actually allocates via 'src_shape', 'src_strides' & 'data_ptr'.
// p.s. If 'src_shape.size()' differs from 'src_strides.size()', it will judge wether 'data_ptr' is null(0-dim but it has value inside) or not.
size_t calcAllocSize(const SizeTArray &src_shape, const SizeTArray &src_strides, const void *data_ptr,
                     const string &file_name, const string &method_name);


// Get the size which actually uses via 'src_shape' & 'data_ptr'.
size_t calcEleSize(const SizeTArray &src_shape, const void *data_ptr);


}; // namespace ndpp::ndpp_data_arch

}; // namespace ndpp