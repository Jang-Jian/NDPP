#pragma once

#include <include/base/Scalar.hpp>
#include <include/tensor/Tensor.hpp>


namespace ndpp
{

// Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.zeros.html
inline Tensor zeros(const IntArray &size, const ndpp_memory::ScalarType stype = ndpp_memory::ScalarType::Float32,
                    const ndpp_memory::DeviceType dtype = ndpp_memory::DeviceType::Host);

// Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.ones.html
inline Tensor onces(const IntArray &size, const ndpp_memory::ScalarType stype = ndpp_memory::ScalarType::Float32,
                    const ndpp_memory::DeviceType dtype = ndpp_memory::DeviceType::Host);

// Creates a tensor of size size filled with fill_value. The tensor’s dtype is inferred from fill_value.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.full.html
inline Tensor full(const Scalar &fill_value, const IntArray &size, 
                   const ndpp_memory::ScalarType scalar_t, const ndpp_memory::DeviceType device_t);



inline Tensor zeros(const IntArray &size, const ndpp_memory::ScalarType scalar_t,
                    const ndpp_memory::DeviceType device_t)
{
    Tensor _dst;
    _dst.zerosB(size, scalar_t, device_t);
    return _dst;
}


inline Tensor onces(const IntArray &size, const ndpp_memory::ScalarType scalar_t,
                    const ndpp_memory::DeviceType device_t)
{
    Tensor _dst;
    _dst.zerosB(size, scalar_t, device_t);

    const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type _one = 1.0f;
    ndpp_memory::memset(_dst.data(), &_one, _dst.scalar(), ndpp_memory::ScalarType::Float32, 
                        _dst.device(), _dst.allocations(), "TensorHelper.hpp", "ndpp::Tensor ndpp::onces()");
    return _dst;
}


inline Tensor full(const Scalar &fill_value, const IntArray &size, 
                   const ndpp_memory::ScalarType scalar_t, const ndpp_memory::DeviceType device_t)
{
    Tensor _dst;
    _dst.zerosB(size, scalar_t, device_t);

    ndpp_memory::memset(_dst.data(), fill_value.data(), _dst.scalar(), fill_value.type(), 
                        _dst.device(), _dst.allocations(), "TensorHelper.hpp", "ndpp::Tensor ndpp::full()");

    return _dst;
}


}; // namespace ndpp.