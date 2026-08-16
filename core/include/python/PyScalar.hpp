#pragma once

#include <boost/shared_ptr.hpp>

#include <include/base/Scalar.hpp>
#include <include/base/DataProc.hpp>


namespace ndpp
{

namespace ndpp_python
{

class PyScalar;
// makePyScalar: Updated Factory Function to handle strict type dispatching.
boost::shared_ptr<PyScalar> makePyScalar(const boost::python::object &data);


class PyScalar : public Scalar
{
public:
    inline PyScalar();
    virtual inline ~PyScalar() = default;
    explicit inline PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type src);
    explicit inline PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src);
    explicit inline PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type src);

    // Get the value via float64.
    inline ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type pydata() const;
};

PyScalar::PyScalar() : Scalar()
{
}

inline PyScalar::PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type src) : Scalar(src)
{
}

inline PyScalar::PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src) : Scalar(src)
{
}

inline PyScalar::PyScalar(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type src) : Scalar(src)
{
}

inline ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type PyScalar::pydata() const
{
    return ndpp_memory::cvtValue<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(data(), type());
}


}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp