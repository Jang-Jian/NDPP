#pragma once

#include <boost/python.hpp>

#include <include/list/List.hpp>
#include <include/base/BasicType.hpp>
#include <include/list/ListHelper.hpp>

using namespace boost::python;


namespace ndpp
{
    
namespace ndpp_python
{

class PyList : public List
{
public:
    PyList() = default;
    inline PyList(List &&src);
    inline PyList(PyList &&src);
    inline PyList(const List &src);
    inline PyList(const PyList &src);

    virtual ~PyList() = default;

    inline void pyinsert(const Integer position, boost::python::object &&object, const ndpp_action::ListPushAction action);
    inline void pyinsert(const Integer position, const boost::python::object &object, const ndpp_action::ListPushAction action);

    inline void pypushback(boost::python::object &&object);
    inline void pypushback(const boost::python::object &object);

    inline void pypushfront(boost::python::object &&object);
    inline void pypushfront(const boost::python::object &object);

    inline void pyerase(const Integer position);

    inline PyList pyclone() const;

    inline void pymigrate(PyList &src);

    // Returns a reference to the first element in the container.
    inline boost::python::object pyfront() const;

    // Returns a reference to the last element in the container.
    inline boost::python::object pyback() const;

    inline void pysplice(const Integer position, PyList &src);
};

inline PyList::PyList(List &&src) : List(std::move(src))
{
}

inline PyList::PyList(PyList &&src) : List(std::move(src))
{
}

inline PyList::PyList(const List &src) : List(src)
{
}

inline PyList::PyList(const PyList &src) : List(src)
{
}

inline void PyList::pyinsert(const Integer position, boost::python::object &&object, const ndpp_action::ListPushAction action)
{
    insert(ndpp::next(begin(), position), std::move(object), action);
}

inline void PyList::pyinsert(const Integer position, const boost::python::object &object, const ndpp_action::ListPushAction action)
{
    insert(ndpp::next(begin(), position), object, action);
}

inline void PyList::pypushback(boost::python::object &&object)
{
    pushBack(std::move(object));
}

inline void PyList::pypushback(const boost::python::object &object)
{
    pushBack(object);
}

inline void PyList::pypushfront(boost::python::object &&object)
{
    pushFront(std::move(object));
}

inline void PyList::pypushfront(const boost::python::object &object)
{
    pushFront(object);
}

inline void PyList::pyerase(const Integer position)
{
    erase(ndpp::next(begin(), position));
}

inline PyList PyList::pyclone() const
{
    List _dst = clone();
    return PyList(std::move(_dst));
}

inline void PyList::pymigrate(PyList &src)
{
    migrate(src);
}

inline boost::python::object PyList::pyfront() const
{
    NodeDevice &_front_node = front();
    return _front_node.data().value<boost::python::object>();
}

inline boost::python::object PyList::pyback() const
{
    NodeDevice &_back_node = back();
    return _back_node.data().value<boost::python::object>();
}

inline void PyList::pysplice(const Integer position, PyList &src)
{
    splice(ndpp::next(begin(), position), src);
}

    
}; // namespace ndpp::ndpp_python

}; // namespace ndpp
