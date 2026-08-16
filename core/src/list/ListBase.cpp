#include <include/list/ListBase.hpp>
#include <include/base/DeviceProc.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_list_base
{

ListDevice::ListDevice()
{
}

ListDevice::ListDevice(ListDevice &&src) : _head(src._head), _tail(src._tail), _num(src._num)
{
    src._head = nullptr;
    src._tail = nullptr;
    src._num = 0;
}

ListDevice::ListDevice(const ListDevice &src)
{
    CopyNodes(src);
}

ListDevice::~ListDevice()
{
    ClearNodes();
}

NodeDevice* ListDevice::CreateNode(Generic &&object) const
{
    NodeDevice *_dst = ndpp_memory::mixMemoryAlloc<NodeDevice>(1, ndpp_memory::DeviceType::Host, 
                                                               "ListBase.cpp", "ListDevice::CreateNode(Generic&&)");

    _dst->_data = new Generic(std::move(object));
    _dst->_prev = nullptr;
    _dst->_next = nullptr;

    return _dst;
}

NodeDevice* ListDevice::CreateNode(const Generic &object) const
{
    NodeDevice *_dst = ndpp_memory::mixMemoryAlloc<NodeDevice>(1, ndpp_memory::DeviceType::Host, 
                                                               "ListBase.cpp", "ListDevice::CreateNode(const Generic&)");

    _dst->_data = new Generic(object);
    _dst->_prev = nullptr;
    _dst->_next = nullptr;

    return _dst;
}

void ListDevice::PushNodeToHead(NodeDevice *src_node)
{
    if (this->_head)
    {
        src_node->_prev = nullptr;
        src_node->_next = this->_head;
        this->_head->_prev = src_node;
    }
    else
    {
        this->_tail = src_node;
    }

    this->_head = src_node;
    ++this->_num;
}

void ListDevice::PushNodeToTail(NodeDevice *src_node)
{
    if (this->_head)
    {
        src_node->_next = nullptr;
        src_node->_prev = this->_tail;
        this->_tail->_next = src_node;
    }
    else
    {
        this->_head = src_node;
    }

    this->_tail = src_node;
    ++this->_num;
}

void ListDevice::InsertNode(NodeDevice *target_node, NodeDevice *src_node, const ndpp_action::ListPushAction action)
{
    // If head=nullptr.
    if (!this->_head)
    {
        PushNodeToHead(src_node);
        return;
    }

    switch (action)
    {
    case ndpp_action::ListPushAction::Front:
        {
            // If the current head is target_node.
            if (target_node == this->_head)
            {
                PushNodeToHead(src_node);
                return;
            }

            // The originial head isn't target_node.
            NodeDevice *prev_node = target_node->_prev;
            src_node->_next = target_node;
            src_node->_prev = prev_node;
            target_node->_prev = src_node;
            prev_node->_next = src_node;
        }
        break;

    case ndpp_action::ListPushAction::Back:
        {
            // If the current tail is target_node.
            if (target_node == this->_tail)
            {
                PushNodeToTail(src_node);
                return;
            }
            
            // The originial tail isn't target_node.
            NodeDevice *next_node = target_node->_next;
            src_node->_next = next_node;
            src_node->_prev = target_node;
            target_node->_next = src_node;
            next_node->_prev = src_node;
        }
        break;
    
    default:
        ndpp_log::logger("ListBase.cpp", "ListDevice::InsertNode()", ndpp_log::RuntimeType::Warn,
                         "There is no matched ndpp_action::ListPushAction.", true);
        return;
    }

    ++this->_num;
}

void ListDevice::EraseNode(NodeDevice *target_node)
{
    if (!this->_head)
    {
        return;
    }

    bool head_tail_same = (this->_head == this->_tail) ? true : false;

    if (!head_tail_same)
    {
        if (this->_head == target_node)
        {
            NodeDevice *_head_tmp = this->_head;
            this->_head = this->_head->_next;
            this->_head->_prev = nullptr;
            delete _head_tmp->_data;
            delete _head_tmp;
        }
        else if (this->_tail == target_node)
        {
            NodeDevice *_tail_tmp = this->_tail;
            this->_tail = this->_tail->_prev;
            this->_tail->_next = nullptr;
            delete _tail_tmp->_data;
            delete _tail_tmp;
        }
        else
        {
            target_node->_prev->_next = target_node->_next;
            target_node->_next->_prev = target_node->_prev;
            delete target_node->_data;
            delete target_node;
        }
    }
    else
    {
        if (this->_head == target_node)
        {
            delete target_node->_data;
            delete target_node;
            this->_head = nullptr;
            this->_tail = nullptr;
        }
    }

    --this->_num;
}

void ListDevice::ClearNodes()
{
    if (this->_head)
    {
        while (this->_head)
        {
            delete this->_head->_data;
            NodeDevice *_head_tmp = this->_head;
            this->_head = this->_head->_next;

            ndpp_memory::mixMemoryDeAlloc<NodeDevice>(_head_tmp, ndpp_memory::DeviceType::Host, 
                                                      "ListBase.cpp", "ListDevice::ClearNodes()");
        }
    }

    this->_head = nullptr;
    this->_tail = nullptr;
    this->_num = 0;
}

void ListDevice::CopyNodes(const ListDevice &src)
{
    if (this == &src)
    {
        return;
    }

    ClearNodes();

    if (!src._head)
    {
        return;
    }

    NodeDevice *_node_tmp = src._head;

    while (_node_tmp)
    {
        NodeDevice *_node_new_cpy = CreateNode(*_node_tmp->_data);
        PushNodeToTail(_node_new_cpy);
        _node_tmp = _node_tmp->_next;
    }
}


}; // namespace ndpp::ndpp_list_base

}; // namespace ndpp