#include <include/python/PyTensor.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_python
{

// getPyTensor: Accessing the PyTensor via row_index from PyTensor.
inline PyTensor const getPyTensor(const PyTensor &src, int64_t row_index);

// getValueFromPyTensor: Accessing the data via row_index from PyTensor.
template<typename T>
inline T const getValueFromPyTensor(const PyTensor &src);

// setPyTensor: Assigning the data 'value' via row_index from PyTensor.
inline void setPyTensor(PyTensor &src, int64_t row_index, const boost::python::object &rhs);


enum class PyOpPyTensor;

inline PyTensor addPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location);
inline PyTensor subPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location);
inline PyTensor mulPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location);
inline PyTensor divPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location);

inline void iAddPyTensor(PyTensor &a, const boost::python::object &b);
inline void iSubPyTensor(PyTensor &a, const boost::python::object &b);
inline void iMulPyTensor(PyTensor &a, const boost::python::object &b);
inline void iDivPyTensor(PyTensor &a, const boost::python::object &b);


inline PyTensor const getPyTensor(const PyTensor &src, int64_t row_index)
{
    return src[row_index];
}
template<typename T>
inline T const getValueFromPyTensor(const PyTensor &src)
{
    return static_cast<T>(src);
}
inline void setPyTensor(PyTensor &src, int64_t row_index, const boost::python::object &rhs)
{
    PyObject* _raw_obj_ptr = rhs.ptr();

    if (PyBool_Check(_raw_obj_ptr))
    {
        src[row_index] = boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(rhs)();
        return;
    }

    if (PyLong_Check(_raw_obj_ptr))
    {
        src[row_index] = boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(rhs)();
        return;
    }

    if (PyFloat_Check(_raw_obj_ptr))
    {
        src[row_index] = boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(rhs)();
        return;
    }

    ndpp_log::logger("PyTensorProc.hpp", "ndpp::ndpp_python::setPyTensor()", ndpp_log::RuntimeType::Error, 
                     "Expected bool, int64 or float64.", true);
    exit(EXIT_FAILURE);
}


struct AddScalarOp
{
    template <typename T>
    inline PyTensor operator()(const PyTensor &lhs, const T &rhs) const
    {
        return lhs + rhs;
    }

    template <typename T>
    inline PyTensor operator()(const T &lhs, const PyTensor &rhs) const
    {
        return lhs + rhs;
    }

    inline PyTensor operator()(const PyTensor &lhs, const PyTensor &rhs) const
    {
        return lhs + rhs;
    }
};
struct SubScalarOp
{
    template <typename T>
    inline PyTensor operator()(const PyTensor& lhs, const T &rhs) const
    {
        return lhs - rhs;
    }

    template <typename T>
    inline PyTensor operator()(const T &lhs, const PyTensor &rhs) const
    {
        return lhs - rhs;
    }

    inline PyTensor operator()(const PyTensor &lhs, const PyTensor &rhs) const
    {
        return lhs - rhs;
    }
};
struct MulScalarOp
{
    template <typename T>
    inline PyTensor operator()(const PyTensor& lhs, const T &rhs) const
    {
        return lhs * rhs;
    }

    template <typename T>
    inline PyTensor operator()(const T &lhs, const PyTensor &rhs) const
    {
        return lhs * rhs;
    }

    inline PyTensor operator()(const PyTensor &lhs, const PyTensor &rhs) const
    {
        return lhs * rhs;
    }
};
struct DivScalarOp
{
    template <typename T>
    inline PyTensor operator()(const PyTensor& lhs, const T &rhs) const
    {
        return lhs / rhs;
    }

    template <typename T>
    inline PyTensor operator()(const T &lhs, const PyTensor &rhs) const
    {
        return lhs / rhs;
    }

    inline PyTensor operator()(const PyTensor &lhs, const PyTensor &rhs) const
    {
        return lhs / rhs;
    }
};


enum class PyOpPyTensor
{
    None      = 0, // a {op} b. 
    Reversed  = 1  // b {op} a.
};


#define DISPATCH_SCALAR_OP(LOCATION, A, B, OP) \
    switch (LOCATION)                          \
    {                                          \
    case PyOpPyTensor::None:                   \
        return OP()(A, B);                     \
    case PyOpPyTensor::Reversed:               \
        return OP()(B, A);                     \
    }


template <typename Op>
inline PyTensor dispatchOp(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{

    boost::python::extract<PyTensor> _b_tensor(b);
    if (_b_tensor.check())
    {
        DISPATCH_SCALAR_OP(location, a, _b_tensor(), Op);
    }

    PyObject* _b_raw_ptr = b.ptr();
    if (PyBool_Check(_b_raw_ptr))
    {
        DISPATCH_SCALAR_OP(location, a, 
                           boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(b)(), Op);
    }
        
    if (PyLong_Check(_b_raw_ptr))
    {
        DISPATCH_SCALAR_OP(location, a, 
                           boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(b)(), Op);
    }

    if (PyFloat_Check(_b_raw_ptr))
    {
        DISPATCH_SCALAR_OP(location, a, 
                           boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(b)(), Op);
    }

    ndpp_log::logger("PyTensorProc.hpp", "ndpp::ndpp_python::dispatchOp()", ndpp_log::RuntimeType::Error, 
                     "Expected tensor, bool, int64 or float64 for b.", true);
    exit(EXIT_FAILURE);

    return PyTensor(); // Never reached, but keeps compiler happy.
}
inline PyTensor addPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{
    return dispatchOp<AddScalarOp>(a, b, location);
}
inline PyTensor subPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{
    return dispatchOp<SubScalarOp>(a, b, location);
}
inline PyTensor mulPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{
    return dispatchOp<MulScalarOp>(a, b, location);
}
inline PyTensor divPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{
    return dispatchOp<DivScalarOp>(a, b, location);
}

#define DISPATCH_FLOOR_OP(LOCATION, A, B, C, OP_NAME) \
    switch (LOCATION)                          \
    {                                          \
    case PyOpPyTensor::None:                   \
        ndpp_arithmetic::arithmeticForward(A, B, C, \
                                           ndpp_arithmetic::Arithmetic::FloorDivision, OP_NAME); \
        break; \
    case PyOpPyTensor::Reversed:               \
        ndpp_arithmetic::arithmeticForward(B, A, C, \
                                           ndpp_arithmetic::Arithmetic::FloorDivision, OP_NAME); \
        break; \
    }

inline PyTensor floorDivPyTensor(const PyTensor &a, const boost::python::object &b, const PyOpPyTensor location)
{
    PyTensor c;
    
    boost::python::extract<PyTensor> _b_tensor(b);
    if (_b_tensor.check())
    {
        DISPATCH_FLOOR_OP(location, a, _b_tensor, c, 
                            "floorDivPyTensor(const PyTensor&, const boost::python::object&, const PyOpPyTensor)");
        return c;
    }


    PyObject* _b_raw_ptr = b.ptr();
    if (PyBool_Check(_b_raw_ptr))
    {
        DISPATCH_FLOOR_OP(location, a, Scalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(b)), c, 
                          "floorDivPyTensor(const PyTensor&, const boost::python::object&, const PyOpPyTensor)");
        return c;
    }
        
    if (PyLong_Check(_b_raw_ptr))
    {
        DISPATCH_FLOOR_OP(location, a, Scalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(b)), c, 
                          "floorDivPyTensor(const PyTensor&, const boost::python::object&, const PyOpPyTensor)");
        return c;
    }

    if (PyFloat_Check(_b_raw_ptr))
    {
        DISPATCH_FLOOR_OP(location, a, Scalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(b)), c, 
                          "floorDivPyTensor(const PyTensor&, const boost::python::object&, const PyOpPyTensor)");
        return c;
    }

    ndpp_log::logger("PyTensorProc.hpp", "ndpp::ndpp_python::floorDivPyTensor()", ndpp_log::RuntimeType::Error, 
                     "Expected tensor, bool, int64 or float64.", true);
    exit(EXIT_FAILURE);

    return PyTensor(); // Never reached, but keeps compiler happy.
}


struct IAddScalarOp
{
    template <typename T>
    inline void operator()(PyTensor &lhs, const T &rhs)
    {
        lhs += rhs;
    }

    inline void operator()(PyTensor &lhs, const PyTensor &rhs)
    {
        lhs += rhs;
    }
};
struct ISubScalarOp
{
    template <typename T>
    inline void operator()(PyTensor &lhs, const T &rhs)
    {
        lhs -= rhs;
    }

    inline void operator()(PyTensor &lhs, const PyTensor &rhs)
    {
        lhs -= rhs;
    }
};
struct IMulScalarOp
{
    template <typename T>
    inline void operator()(PyTensor &lhs, const T &rhs)
    {
        lhs *= rhs;
    }

    inline void operator()(PyTensor &lhs, const PyTensor &rhs)
    {
        lhs *= rhs;
    }
};
struct IDivScalarOp
{
    template <typename T>
    inline void operator()(PyTensor &lhs, const T &rhs)
    {
        lhs /= rhs;
    }

    inline void operator()(PyTensor &lhs, const PyTensor &rhs)
    {
        lhs /= rhs;
    }
};

template <typename Op>
inline void dispatchIOp(PyTensor &a, const boost::python::object &b)
{
    boost::python::extract<PyTensor> _b_tensor(b);
    if (_b_tensor.check())
    {
        Op()(a, _b_tensor());
        return;
    }

    PyObject* _b_raw_ptr = b.ptr();
    if (PyBool_Check(_b_raw_ptr))
    {
        Op()(a, boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(b)());
        return;
    }
        
    if (PyLong_Check(_b_raw_ptr))
    {
        Op()(a, boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(b)());
        return;
    }

    if (PyFloat_Check(_b_raw_ptr))
    {
        Op()(a, boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(b)());
        return;
    }
   
    ndpp_log::logger("PyTensorProc.hpp", "ndpp::ndpp_python::dispatchIOp()", ndpp_log::RuntimeType::Error, 
                     "Expected bool, int64 or float64.", true);
}
inline void iAddPyTensor(PyTensor &a, const boost::python::object &b)
{
    dispatchIOp<IAddScalarOp>(a, b);
}
inline void iSubPyTensor(PyTensor &a, const boost::python::object &b)
{
    dispatchIOp<ISubScalarOp>(a, b);
}
inline void iMulPyTensor(PyTensor &a, const boost::python::object &b)
{
    dispatchIOp<IMulScalarOp>(a, b);
}
inline void iDivPyTensor(PyTensor &a, const boost::python::object &b)
{
    dispatchIOp<IDivScalarOp>(a, b);
}


}; // namespace ndpp::ndpp_python

}; // namespace ndpp