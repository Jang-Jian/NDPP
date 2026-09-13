#pragma once

#include <include/python/PyStack.hpp>


namespace ndpp
{

namespace ndpp_python
{

inline void iStackPush(PyStack &self, boost::python::object &object)
{
    self.pypush(std::move(object));
}

}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp