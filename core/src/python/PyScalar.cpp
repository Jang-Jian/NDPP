#include <boost/shared_ptr.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/make_constructor.hpp>

#include <include/python/PyScalar.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_python
{

boost::shared_ptr<PyScalar> makePyScalar(const boost::python::object &data) 
{
    void *_raw_ptr = data.ptr();

    // Explicitly check for Python Bool first (since Python bool inherits from int)
    if (PyBool_Check(_raw_ptr)) 
    {
        return boost::shared_ptr<PyScalar>(new PyScalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(data)));
    } 
    else if (PyLong_Check(_raw_ptr)) 
    { 
        return boost::shared_ptr<PyScalar>(new PyScalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(data)));
    }
    else if (PyFloat_Check(_raw_ptr)) 
    {
        return boost::shared_ptr<PyScalar>(new PyScalar(boost::python::extract<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(data)));
    } 
    
    // Throw an error back to Python if the type does not match either
    ndpp_log::logger("PyScalar.cpp", "ndpp::ndpp_python::makePyScalar()", ndpp_log::RuntimeType::Warn, 
                     "ndpp::ndpp_python::PyScalar's constructor requires either an 'int', a 'float' or a 'bool' type, and it will return default float64 constructor.", true);
    return boost::shared_ptr<PyScalar>(new PyScalar());
}


}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp