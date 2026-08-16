#pragma once

#include <cstdint>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <include/tensor/Tensor.hpp>
#include <include/python/PyScalar.hpp>
#include <include/logging/Logging.hpp>

using namespace boost::python;


namespace ndpp
{

namespace ndpp_python
{

class PyTensor : public Tensor
{
public:
    PyTensor() = default;
    inline PyTensor(Tensor &&src);
    inline PyTensor(PyTensor &&src);
    inline PyTensor(const Tensor &src);
    inline PyTensor(const PyTensor &src);

    virtual ~PyTensor() = default;


    inline PyTensor operator[](int64_t index) const;

    template<typename T>
    inline PyTensor& operator=(const T src);

    template<typename T>
    inline PyTensor& operator+=(const T         b); // PyTensor += scalar(template).
    inline PyTensor& operator+=(const PyScalar &b); // PyTensor += PyScalar.
    inline PyTensor& operator+=(const PyTensor &b); // PyTensor += PyTensor.

    template<typename T>
    inline PyTensor& operator-=(const T         b); // PyTensor -= scalar(template).
    inline PyTensor& operator-=(const PyScalar &b); // PyTensor -= PyScalar.
    inline PyTensor& operator-=(const PyTensor &b); // PyTensor -= PyTensor.

    template<typename T>
    inline PyTensor& operator*=(const T         b); // PyTensor *= scalar(template).
    inline PyTensor& operator*=(const PyScalar &b); // PyTensor *= PyScalar.
    inline PyTensor& operator*=(const PyTensor &b); // PyTensor *= PyTensor.

    template<typename T>
    inline PyTensor& operator/=(const T         b); // PyTensor /= scalar(template).
    inline PyTensor& operator/=(const PyScalar &b); // PyTensor /= PyScalar.
    inline PyTensor& operator/=(const PyTensor &b); // PyTensor /= PyTensor.


    template<typename T>
    friend inline PyTensor operator+(const PyTensor &a, const T         b); // PyTensor + scalar(template).
    template<typename T>
    friend inline PyTensor operator+(const T         a, const PyTensor &b); // scalar(template) + PyTensor.
    friend inline PyTensor operator+(const PyTensor &a, const PyScalar &b); // PyTensor + PyScalar.
    friend inline PyTensor operator+(const PyScalar &a, const PyTensor &b); // PyScalar + PyTensor.
    friend inline PyTensor operator+(const PyTensor &a, const PyTensor &b); // PyTensor + PyTensor.

    template<typename T>
    friend inline PyTensor operator-(const PyTensor &a, const T         b); // PyTensor - scalar(template).
    template<typename T>
    friend inline PyTensor operator-(const T         a, const PyTensor &b); // scalar(template) - PyTensor.
    friend inline PyTensor operator-(const PyTensor &a, const PyScalar &b); // PyTensor - PyScalar.
    friend inline PyTensor operator-(const PyScalar &a, const PyTensor &b); // PyScalar - PyTensor.
    friend inline PyTensor operator-(const PyTensor &a, const PyTensor &b); // PyTensor - PyTensor.

    template<typename T>
    friend inline PyTensor operator*(const PyTensor &a, const T         b); // PyTensor * scalar(template).
    template<typename T>
    friend inline PyTensor operator*(const T         a, const PyTensor &b); // scalar(template) * PyTensor.
    friend inline PyTensor operator*(const PyTensor &a, const PyScalar &b); // PyTensor * Scalar.
    friend inline PyTensor operator*(const PyScalar &a, const PyTensor &b); // Scalar * PyTensor.
    friend inline PyTensor operator*(const PyTensor &a, const PyTensor &b); // PyTensor * PyTensor.

    template<typename T>
    friend inline PyTensor operator/(const PyTensor &a, const T         b); // PyTensor / scalar(template).
    template<typename T>
    friend inline PyTensor operator/(const T         a, const PyTensor &b); // scalar(template) / PyTensor.
    friend inline PyTensor operator/(const PyTensor &a, const PyScalar &b); // PyTensor / PyScalar.
    friend inline PyTensor operator/(const PyScalar &a, const PyTensor &b); // PyScalar / PyTensor.
    friend inline PyTensor operator/(const PyTensor &a, const PyTensor &b); // PyTensor / PyTensor.


    // Get the shape.
    boost::python::tuple pysizes() const;

    // Get the strides.
    boost::python::tuple pystrides() const;

    inline void pycopy(const PyTensor &src, const ndpp_memory::DeviceType dtype);

    inline PyTensor pyclone(const ndpp_memory::DeviceType dtype) const;

    inline PyTensor pyto(const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype) const;

    // Migrates data from PyTensor.
    inline void pymigrate(PyTensor &src);

    // Copy data from PyTensor.
    void pyfromtensor(const PyTensor &src,
                      const ndpp_memory::DeviceType dst_dtype,
                      const ndpp_memory::DeviceStatus copy_status);

    // Copy data from boost::python::numpy::ndarray.
    // P.S src_dtype is used for src's DeviceType, dst_dtype is used for destination (PyTensor) and with DeviceStatus::Allocation only.
    void pyfromnumpy(const numpy::ndarray &src,
                     const ndpp_memory::DeviceType src_dtype,
                     const ndpp_memory::DeviceType dst_dtype,
                     const ndpp_memory::DeviceStatus copy_status);
};

inline PyTensor::PyTensor(Tensor &&src) : Tensor(std::move(src))
{
}

inline PyTensor::PyTensor(PyTensor &&src) : Tensor(std::move(src))
{
}

inline PyTensor::PyTensor(const Tensor &src) : Tensor(src)
{
}

inline PyTensor::PyTensor(const PyTensor &src) : Tensor(src)
{
}

inline PyTensor PyTensor::operator[](int64_t index) const
{
    return PyTensor(std::move(Tensor::operator[](index)));
}

template<typename T>
inline PyTensor& PyTensor::operator=(const T src)
{
    Tensor::operator=(src);
    return *this;
}

template<typename T>
inline PyTensor& PyTensor::operator+=(const T b)
{
    Tensor::operator+=(b);
    return *this;
}

inline PyTensor& PyTensor::operator+=(const PyScalar &b)
{
    Tensor::operator+=(b);
    return *this;
}
    
inline PyTensor& PyTensor::operator+=(const PyTensor &b)
{
    Tensor::operator+=(static_cast<const Tensor&>(b));
    return *this;
}

template<typename T>
inline PyTensor& PyTensor::operator-=(const T b)
{
    Tensor::operator-=(b);
    return *this;
}

inline PyTensor& PyTensor::operator-=(const PyScalar &b)
{
    Tensor::operator-=(b);
    return *this;
}

inline PyTensor& PyTensor::operator-=(const PyTensor &b)
{
    Tensor::operator-=(static_cast<const Tensor&>(b));
    return *this;
}

template<typename T>
inline PyTensor& PyTensor::operator*=(const T b)
{
    Tensor::operator*=(b);
    return *this;
}

inline PyTensor& PyTensor::operator*=(const PyScalar &b)
{
    Tensor::operator*=(b);
    return *this;
}

inline PyTensor& PyTensor::operator*=(const PyTensor &b)
{
    Tensor::operator*=(static_cast<const Tensor&>(b));
    return *this;
}

template<typename T>
inline PyTensor& PyTensor::operator/=(const T b)
{
    Tensor::operator/=(b);
    return *this;
}

inline PyTensor& PyTensor::operator/=(const PyScalar &b)
{
    Tensor::operator/=(b);
    return *this;
}

inline PyTensor& PyTensor::operator/=(const PyTensor &b)
{
    Tensor::operator/=(static_cast<const Tensor&>(b));
    return *this;
}

template<typename T>
inline PyTensor operator+(const PyTensor &a, const T b)
{
    return PyTensor(std::move(operator+(static_cast<const Tensor&>(a), b)));
}

template<typename T>
inline PyTensor operator+(const T a, const PyTensor &b)
{
    return PyTensor(std::move(operator+(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator+(const PyTensor &a, const PyScalar &b)
{
    return PyTensor(std::move(operator+(static_cast<const Tensor&>(a), b)));
}

inline PyTensor operator+(const PyScalar &a, const PyTensor &b)
{
    return PyTensor(std::move(operator+(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator+(const PyTensor &a, const PyTensor &b)
{
    return PyTensor(std::move(operator+(static_cast<const Tensor&>(a), static_cast<const Tensor&>(b))));
}

template<typename T>
inline PyTensor operator-(const PyTensor &a, const T b)
{
    return PyTensor(std::move(operator-(static_cast<const Tensor&>(a), b)));
}

template<typename T>
inline PyTensor operator-(const T a, const PyTensor &b)
{
    return PyTensor(std::move(operator-(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator-(const PyTensor &a, const PyScalar &b)
{
    return PyTensor(std::move(operator-(static_cast<const Tensor&>(a), b)));
}

inline PyTensor operator-(const PyScalar &a, const PyTensor &b)
{
    return PyTensor(std::move(operator-(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator-(const PyTensor &a, const PyTensor &b)
{
    return PyTensor(std::move(operator-(static_cast<const Tensor&>(a), static_cast<const Tensor&>(b))));
}

template<typename T>
inline PyTensor operator*(const PyTensor &a, const T b)
{
    return PyTensor(std::move(operator*(static_cast<const Tensor&>(a), b)));
}

template<typename T>
inline PyTensor operator*(const T a, const PyTensor &b)
{
    return PyTensor(std::move(operator*(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator*(const PyTensor &a, const PyScalar &b)
{
    return PyTensor(std::move(operator*(static_cast<const Tensor&>(a), b)));
}

inline PyTensor operator*(const PyScalar &a, const PyTensor &b)
{
    return PyTensor(std::move(operator*(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator*(const PyTensor &a, const PyTensor &b)
{
    return PyTensor(std::move(operator*(static_cast<const Tensor&>(a), static_cast<const Tensor&>(b))));
}

template<typename T>
inline PyTensor operator/(const PyTensor &a, const T b)
{
    return PyTensor(std::move(operator/(static_cast<const Tensor&>(a), b)));
}

template<typename T>
inline PyTensor operator/(const T a, const PyTensor &b)
{
    return PyTensor(std::move(operator/(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator/(const PyTensor &a, const PyScalar &b)
{
    return PyTensor(std::move(operator/(static_cast<const Tensor&>(a), b)));
}

inline PyTensor operator/(const PyScalar &a, const PyTensor &b)
{
    return PyTensor(std::move(operator/(a, static_cast<const Tensor&>(b))));
}

inline PyTensor operator/(const PyTensor &a, const PyTensor &b)
{
    return PyTensor(std::move(operator/(static_cast<const Tensor&>(a), static_cast<const Tensor&>(b))));
}

inline void PyTensor::pycopy(const PyTensor &src, const ndpp_memory::DeviceType dtype)
{
    copy(src, dtype);
}

inline PyTensor PyTensor::pyclone(const ndpp_memory::DeviceType dtype) const
{
    Tensor _dst = clone(dtype);
    return PyTensor(std::move(_dst));
}

inline PyTensor PyTensor::pyto(const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype) const
{
    Tensor _dst = to(stype, dtype);
    return PyTensor(std::move(_dst));
}

inline void PyTensor::pymigrate(PyTensor &src)
{
    migrate(src);
}


}; // namespace ndpp::ndpp_python

}; // namespace ndpp