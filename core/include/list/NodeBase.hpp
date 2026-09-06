#pragma once

#include <include/base/GenericBase.hpp>


namespace ndpp
{

struct NodeDevice;

namespace ndpp_list_base
{

class ListDevice;

}; // namespace ndpp::ndpp_list_base


namespace ndpp_python
{

class PyListIterator;

};


/*
    NodeDevice: The node which saves generic data type, and connecting previous NodeDevice & next NodeDevice.
    P.S Node's Structure.

    | (Previous) (Generic Data) (Next)  |
*/
struct NodeDevice
{
public:
    NodeDevice() = default;

    // Get Generic object which is saved in current node.
    inline Generic& data() const;

    // Get previous NodeDevice which connect with current NodeDevice.
    inline NodeDevice& prevNode() const;

    // Get next NodeDevice which connect with current NodeDevice.
    inline NodeDevice& nextNode() const;

private:
    Generic *_data = nullptr;
    NodeDevice *_prev = nullptr;
    NodeDevice *_next = nullptr;

    // P.S Used for ndpp::List & ndpp::ndpp_list_base::List to easily access.
    friend class List;
    friend class ListIterator;
    friend class ndpp_list_base::ListDevice;
};

inline Generic& NodeDevice::data() const
{
    return *this->_data;
}

inline NodeDevice& NodeDevice::prevNode() const
{
    return *this->_prev;
}

inline NodeDevice& NodeDevice::nextNode() const
{
    return *this->_next;
}


}; // namespace ndpp