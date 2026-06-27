#include <include/list/List.hpp>


namespace ndpp
{

void List::splice(ListIterator position, List &src)
{
    if (!src._head)
    {
        return;
    }

    if (this == &src)
    {
        return;
    }
    
    NodeDevice *_target_node = &(*position);

    if (_target_node)
    {
        // Original list has the size over 1.
        if (this->_tail == _target_node)
        {
            // If target_node is in tail.
            _target_node->_next = src._head;
            src._head->_prev = _target_node;
            this->_tail = src._tail;
        }
        else
        {
            // If target_node isn't in tail.
            NodeDevice *_target_next = _target_node->_next;
            
            _target_node->_next = src._head;
            src._head->_prev = _target_node;

            src._tail->_next = _target_next;
            _target_next->_prev = src._tail;
        }

        this->_num += src._num;
    }
    else
    {
        // Original list is empty.
        this->_head = src._head;
        this->_tail = src._tail;
        this->_num = src._num;
    }

    src._head = nullptr;
    src._tail = nullptr;
    src._num = 0;
}

}; // namespace ndpp