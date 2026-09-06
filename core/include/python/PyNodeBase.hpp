#pragma once

#include <boost/python.hpp>

#include <include/list/NodeBase.hpp>

using namespace boost::python;


namespace ndpp
{

namespace ndpp_python
{


class PyNodeDevice
{
public:
    inline PyNodeDevice() : _node_ptr(nullptr) {}
    inline PyNodeDevice(NodeDevice &node) : _node_ptr(&node) {}

    inline bool pyexists() const
    {
        return this->_node_ptr ? true : false;
    }

    inline boost::python::object pydata() const
    {
        if (!this->_node_ptr) return boost::python::object(); 
        return _node_ptr->data().value<boost::python::object>();
    }

    inline PyNodeDevice pyprevnode() const
    {
        if (!this->_node_ptr) return PyNodeDevice();
        return PyNodeDevice(_node_ptr->prevNode());
    }

    inline PyNodeDevice pynextnode() const
    {
        if (!this->_node_ptr) return PyNodeDevice();
        return PyNodeDevice(_node_ptr->nextNode());
    }

private:
    NodeDevice* _node_ptr; 
};

};

};