#pragma once

#include <include/python/PyQueue.hpp>


namespace ndpp
{

namespace ndpp_python
{

inline void iQueuePush(PyQueue &self, boost::python::object &object)
{
    self.pypush(std::move(object));
}

}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp