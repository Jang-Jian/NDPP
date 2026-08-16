#include <include/base/ScalarType.hpp>
#include <include/python/PyTensor.hpp>


namespace ndpp
{

namespace ndpp_python
{


static inline const IntArray getNumpyShape(const numpy::ndarray &src)
{
    IntArray _shape;
    _shape.refer(const_cast<Integer*>(reinterpret_cast<const Integer*>(src.get_shape())), // origin type: Py_intptr_t.
                 static_cast<size_t>(src.get_nd()), 
                 ndpp_memory::DeviceType::Host);
    return _shape;
}

static inline const IntArray getNumpyStrides(const numpy::ndarray &src)
{
    IntArray _stride;
    _stride.refer(const_cast<Integer*>(reinterpret_cast<const Integer*>(src.get_strides())), // origin type: Py_intptr_t.
                  static_cast<size_t>(src.get_nd()), 
                  ndpp_memory::DeviceType::Host);
    return _stride;
}

static inline const string getNumpyScalarTypeStr(const numpy::ndarray &src)
{
    return boost::python::extract<string>(boost::python::str(src.get_dtype()));
}



boost::python::tuple PyTensor::pysizes() const
{
    const IntArray _ref_shape = sizes();
    const size_t _dim = _ref_shape.size();

    PyObject* _tup_shape = PyTuple_New(_dim);

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(_dim); ++i)
    {
        PyTuple_SET_ITEM(
            _tup_shape,
            i,
            PyLong_FromLongLong(_ref_shape[i]));   // steals the reference
    }

    return boost::python::tuple(boost::python::handle<>(_tup_shape));
}

boost::python::tuple PyTensor::pystrides() const
{
    const IntArray _ref_strides = strides();
    const size_t _dim = _ref_strides.size();

    PyObject* _tup_strides = PyTuple_New(_dim);

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(_dim); ++i)
    {
        PyTuple_SET_ITEM(
            _tup_strides,
            i,
            PyLong_FromLongLong(_ref_strides[i]));   // steals the reference
    }

    return boost::python::tuple(boost::python::handle<>(_tup_strides));
}

void PyTensor::pyfromtensor(const PyTensor &src,
                            const ndpp_memory::DeviceType dst_dtype,
                            const ndpp_memory::DeviceStatus copy_status)
{
    const IntArray _src_shape = src.sizes();
    const IntArray _src_stride = src.strides();

    const ndpp_memory::ScalarType _src_stype = src.scalar();
    const ndpp_memory::DeviceType _src_dtype = src.device();
    void* _raw_ptr = src.data();

    switch (copy_status)
    {
    case ndpp_memory::DeviceStatus::Allocation:
        {
            Tensor::copy(_raw_ptr, _src_shape, _src_stride, 
                         _src_stype, _src_dtype, dst_dtype);
        }
        break;
    
    case ndpp_memory::DeviceStatus::Reference:
        {
            Tensor::refer(_raw_ptr, _src_shape, _src_stride, 
                          _src_stype, _src_dtype);
        }
        break;
    }
}

void PyTensor::pyfromnumpy(const numpy::ndarray &src,
                           const ndpp_memory::DeviceType src_dtype,
                           const ndpp_memory::DeviceType dst_dtype,
                           const ndpp_memory::DeviceStatus copy_status)
{
    // here has errors ... must be checked ...
    ndpp_memory::ScalarType _src_stype = ndpp_memory::cppStrToScalarType(getNumpyScalarTypeStr(src), true);

    const IntArray _src_shape = getNumpyShape(src);
    const IntArray _src_stride = getNumpyStrides(src);

    void* _raw_ptr = static_cast<void*>(src.get_data());
    
    switch (copy_status)
    {
    case ndpp_memory::DeviceStatus::Allocation:
        {
            Tensor::copy(_raw_ptr, _src_shape, _src_stride, 
                         _src_stype, src_dtype, dst_dtype);
        }
        break;
    
    case ndpp_memory::DeviceStatus::Reference:
        {
            Tensor::refer(_raw_ptr, _src_shape, _src_stride, 
                          _src_stype, src_dtype);
        }
        break;
    }
}


}; // namespace ndpp::ndpp_python

}; // namespace ndpp