#pragma once

#include <boost/python.hpp>

#include <include/list/List.hpp>
#include <include/base/BasicType.hpp>
#include <include/list/ListHelper.hpp>
#include <include/python/PyNodeBase.hpp>
#include <include/iterator/PyBasicIterator.hpp>

using namespace boost::python;


namespace ndpp
{
    
namespace ndpp_python
{


class PyListIterator : public PyBasicIterator<NodeDevice, PyNodeDevice, PyListIterator>
{
public:
    inline PyListIterator(NodeDevice *ptr) : PyBasicIterator<NodeDevice, PyNodeDevice, PyListIterator>(ptr) {}

    inline PyNodeDevice operator*() const override
    { 
        return PyNodeDevice(*this->_t_ptr); 
    }

    inline PyListIterator& operator++() override
    {
        this->_t_ptr = &this->_t_ptr->nextNode();
        return *this;
    }

    inline PyListIterator& operator--() override
    {
        this->_t_ptr = &this->_t_ptr->prevNode();
        return *this;
    }

    inline PyListIterator operator++(int) override
    {
        PyListIterator old(*this);
        ++(*this);
        return old;
    }

    inline PyListIterator& pyiter() override
    {
        return *this;
    }

    inline PyNodeDevice pynext() override
    {
        if (_t_ptr == nullptr)
        {
            PyErr_SetNone(PyExc_StopIteration);
            boost::python::throw_error_already_set();
        }

        PyNodeDevice result = **this;
        ++(*this);

        return result;
    }
};


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


    inline PyListIterator pybegin() const
    {
        return PyListIterator(&*begin());
    }

    inline PyListIterator pyend() const
    {
        return PyListIterator(&*end());
    }

    // Returns a reference to the first element in the container.
    inline PyNodeDevice pyfront() const;

    // Returns a reference to the last element in the container.
    inline PyNodeDevice pyback() const;

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

inline PyNodeDevice PyList::pyfront() const
{
    return PyNodeDevice(front());
}

inline PyNodeDevice PyList::pyback() const
{
    return PyNodeDevice(back());
}

inline void PyList::pysplice(const Integer position, PyList &src)
{
    splice(ndpp::next(begin(), position), src);
}

    
}; // namespace ndpp::ndpp_python

}; // namespace ndpp
