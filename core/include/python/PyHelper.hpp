#pragma once

#include <boost/python.hpp>

#include <include/python/PyScalar.hpp>
#include <include/python/PyTensor.hpp>


namespace ndpp
{

namespace ndpp_python
{

inline IntArray cvtSizes(const boost::python::tuple &src)
{
    size_t length = static_cast<size_t>(boost::python::len(src));

    IntArray dst;
    dst.allocate(length, ndpp_memory::DeviceType::Host);

    for (size_t index = 0; index < length; ++index)
    {
        dst[index] = boost::python::extract<Integer>(src[index]);
    }

    return dst;
}


inline PyTensor pyzeros(const boost::python::tuple &size, const ndpp_memory::ScalarType scalar_t, 
                        const ndpp_memory::DeviceType device_t)
{
    IntArray _size = cvtSizes(size);
    return PyTensor(std::move(zeros(_size, scalar_t, device_t)));
}


inline PyTensor pyones(const boost::python::tuple &size, const ndpp_memory::ScalarType scalar_t,
                       const ndpp_memory::DeviceType device_t)
{
    IntArray _size = cvtSizes(size);
    return PyTensor(std::move(ones(_size, scalar_t, device_t)));
}


inline PyTensor pyfull(const PyScalar &fill_value, const boost::python::tuple &size, 
                       const ndpp_memory::ScalarType scalar_t, const ndpp_memory::DeviceType device_t)
{
    IntArray _size = cvtSizes(size);
    return PyTensor(std::move(full(fill_value, _size, scalar_t, device_t)));
}
    
}; // namespace ndpp::ndpp_python

}; // namespace ndpp