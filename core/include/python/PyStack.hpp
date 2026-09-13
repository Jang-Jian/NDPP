#pragma once

#include <boost/python.hpp>

#include <include/segment/Stack.hpp>
#include <include/python/PyNodeBase.hpp>

using namespace boost::python;


namespace ndpp
{

namespace ndpp_python
{

class PyStack : public Stack
{
public:
    PyStack() = default;
    inline PyStack(Stack &&src);
    inline PyStack(PyStack &&src);
    inline PyStack(const Stack &src);
    inline PyStack(const PyStack &src);

    virtual ~PyStack() = default;
    
    inline PyStack pyclone() const;

    inline void pymovefrom(PyStack &src);

    inline void pypush(boost::python::object &&object);
    inline void pypush(const boost::python::object &object);

    inline boost::python::object pytop() const;
};

inline PyStack::PyStack(Stack &&src) : Stack(std::move(src))
{
}

inline PyStack::PyStack(PyStack &&src) : Stack(std::move(src))
{
}
    
inline PyStack::PyStack(const Stack &src) : Stack(src)
{
}

inline PyStack::PyStack(const PyStack &src) : Stack(src)
{
}

inline PyStack PyStack::pyclone() const
{
    Stack _dst = clone();
    return PyStack(std::move(_dst));
}

inline void PyStack::pymovefrom(PyStack &src)
{
    moveFrom(src);
}

inline void PyStack::pypush(boost::python::object &&object)
{
    push(std::move(object));
}

inline void PyStack::pypush(const boost::python::object &object)
{
    push(object);
}

inline boost::python::object PyStack::pytop() const
{
    auto &top_node = top();
    return top_node.value<boost::python::object>();
}


};

};