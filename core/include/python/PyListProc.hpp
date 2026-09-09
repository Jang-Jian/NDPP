#pragma once

#include <include/python/PyList.hpp>


namespace ndpp
{

namespace ndpp_python
{

inline void iListInsert(PyList &self, const Integer index, boost::python::object &object,
                        const ndpp_action::ListPushAction action)
{
    self.pyinsert(index, std::move(object), action);
}

inline void iListPushBack(PyList &self, boost::python::object &object)
{
    self.pypushback(std::move(object));
}

inline void iListPushFront(PyList &self, boost::python::object &object)
{
    self.pypushfront(std::move(object));
}

}; // namespace ndpp::ndpp_python
    
}; // namespace ndpp