#pragma once

#include <cstddef>

#include <include/list/NodeBase.hpp>
#include <include/base/GenericBase.hpp>
#include <include/list/ListPushAction.hpp>


namespace ndpp
{

namespace ndpp_list_base
{


/*
    ListDevice: Used for ndpp::List, which processes memory processing.

    P.S The default allocation ListDevice() has these the settings in below.
    _head = nullptr.
    _tail = nullptr.
    _num = 0.
*/
class ListDevice
{
public:
    ListDevice();
    ListDevice(ListDevice &&src);
    ListDevice(const ListDevice &src);

    virtual ~ListDevice();

protected:
    inline const size_t NumNodes() const;

    NodeDevice* CreateNode(Generic &&object) const;

    NodeDevice* CreateNode(const Generic &object) const;

    void PushNodeToHead(NodeDevice *src_node);

    void PushNodeToTail(NodeDevice *src_node);

    void InsertNode(NodeDevice *target_node, NodeDevice *src_node, const ndpp_action::ListPushAction action);

    void EraseNode(NodeDevice *target_node);

    void ClearNodes();

    void CopyNodes(const ListDevice &src);

    // P.S Due to more faster access, we record the final node address.
    NodeDevice *_head = nullptr;
    NodeDevice *_tail = nullptr;
    size_t _num = 0;
};

inline const size_t ListDevice::NumNodes() const
{
    return this->_num;
}


}; // namespace ndpp::ndpp_list_base

}; // namespace ndpp