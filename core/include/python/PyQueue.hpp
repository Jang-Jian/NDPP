#pragma once

#include <boost/python.hpp>

#include <include/segment/Queue.hpp>
#include <include/python/PyNodeBase.hpp>

using namespace boost::python;


namespace ndpp
{

namespace ndpp_python
{

class PyQueue : public Queue
{
public:
    PyQueue() = default;
    inline PyQueue(Queue &&src);
    inline PyQueue(PyQueue &&src);
    inline PyQueue(const Queue &src);
    inline PyQueue(const PyQueue &src);

    virtual ~PyQueue() = default;
    
    inline PyQueue pyclone() const;

    inline void pymovefrom(PyQueue &src);

    inline void pypush(boost::python::object &&object);
    inline void pypush(const boost::python::object &object);

    inline boost::python::object pyfront() const;

    inline boost::python::object pyback() const;
};

inline PyQueue::PyQueue(Queue &&src) : Queue(std::move(src))
{
}

inline PyQueue::PyQueue(PyQueue &&src) : Queue(std::move(src))
{
}
    
inline PyQueue::PyQueue(const Queue &src) : Queue(src)
{
}

inline PyQueue::PyQueue(const PyQueue &src) : Queue(src)
{
}

inline PyQueue PyQueue::pyclone() const
{
    Queue _dst = clone();
    return PyQueue(std::move(_dst));
}

inline void PyQueue::pymovefrom(PyQueue &src)
{
    moveFrom(src);
}

inline void PyQueue::pypush(boost::python::object &&object)
{
    push(std::move(object));
}

inline void PyQueue::pypush(const boost::python::object &object)
{
    push(object);
}

inline boost::python::object PyQueue::pyfront() const
{
    auto &first_node = front();
    return first_node.value<boost::python::object>();
}

inline boost::python::object PyQueue::pyback() const
{
    auto &final_node = back();
    return final_node.value<boost::python::object>();
}


};

};