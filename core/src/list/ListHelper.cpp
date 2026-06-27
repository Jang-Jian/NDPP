#include <include/list/List.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

ListIterator next(ListIterator src, const size_t steps)
{
    NodeDevice *_node = &(*src);
    if (!_node)
    {
        return ListIterator(nullptr);
    }

    for (size_t index = 0; index < steps; ++index)
    {
        NodeDevice *_next = &_node->nextNode();
        if (!_next)
        {
            break;
        }
        else
        {
            _node =_next;
        }
    }

    return ListIterator(_node);
}

}; // namespace ndpp