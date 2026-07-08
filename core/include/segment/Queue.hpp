#pragma once

#include <include/list/List.hpp>


namespace ndpp
{

/*
    Queue: A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    P.S Inheriting the ndpp::List to do FIFO / LILO.
    Reference: https://en.cppreference.com/cpp/container/queue
*/
class Queue : private List
{
public:
    Queue() = default;
    inline Queue(Queue &&src);
    inline Queue(const Queue &src);

    virtual ~Queue() = default;

    /* 
        Used for the specified situation.
        Queue dst;
        dst = func(...);
    */
    inline Queue& operator=(const Queue &src);

    // Clone itself to new Queue.
    inline Queue clone() const;

    // Returns the number of elements.
    inline const size_t size() const;

    // Checks whether the container adaptor is empty.
    inline const bool empty() const;

    // Pushes the given element value to the end of the queue. (used for move constructor).
    inline void push(Generic &&object);

    // Pushes the given element value to the end of the queue. (used for copy constructor).
    inline void push(const Generic &object);

    // Accesses the first element.
    inline Generic& front() const;

    // Accesses the last element.
    inline Generic& back() const;

    // Removes the top element.
    inline void pop();
};

inline Queue::Queue(Queue &&src) : List(std::move(src))
{
}

inline Queue::Queue(const Queue &src) : List(src)
{
}

inline Queue& Queue::operator=(const Queue &src)
{
    List::operator=(src);
    return *this;
}

inline Queue Queue::clone() const
{
    Queue _dst;
    _dst.copy(*this);
    return _dst; 
}

inline const size_t Queue::size() const
{
    return List::size();
}

inline const bool Queue::empty() const
{
    return (List::size() > 0) ? false : true;
}

inline void Queue::push(Generic &&object)
{
    pushBack(std::move(object));
}

inline void Queue::push(const Generic &object)
{
    pushBack(object);
}

inline Generic& Queue::front() const
{
    auto &first_node = List::front();
    return first_node.data();
}

inline Generic& Queue::back() const
{
    auto &final_node = List::back();
    return final_node.data();
}

inline void Queue::pop()
{
    erase(&List::front());
}

}; // namespace ndpp