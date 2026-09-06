#pragma once

#include <include/python/PyList.hpp>


namespace ndpp
{

namespace ndpp_python
{

inline void iInsert(PyList &self, const Integer index, boost::python::object &object,
                    const ndpp_action::ListPushAction action)
{
    self.pyinsert(index, std::move(object), action);
}

inline void iPushBack(PyList &self, boost::python::object &object)
{
    self.pypushback(std::move(object));
}

inline void iPushFront(PyList &self, boost::python::object &object)
{
    self.pypushfront(std::move(object));
}

}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp