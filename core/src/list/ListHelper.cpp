#include <cmath>

#include <include/list/List.hpp>
#include <include/logging/Logging.hpp>

using namespace std;


namespace ndpp
{


void advance(ListIterator &src, const std::ptrdiff_t steps)
{
    NodeDevice *_node = &(*src);
    if (!_node)
    {
        src = ListIterator(nullptr);
    }

    if (steps == 0)
    {
        return;
    }

    if (steps > 0)
    {
        for (std::ptrdiff_t index = 0; index < steps; ++index)
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
    }
    else
    {
        for (std::ptrdiff_t index = 0; index < std::abs(steps); ++index)
        {
            NodeDevice *_prev = &_node->prevNode();
            if (!_prev)
            {
                break;
            }
            else
            {
                _node = _prev;
            }
        }
    }

    src = ListIterator(_node);
}


}; // namespace ndpp