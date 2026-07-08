#pragma once

#include <include/list/List.hpp>


namespace ndpp
{

/*
    Stack: A LIFO (Last-In, First-Out) / FILO (First-In, Last-Out) data structure.
    P.S Inheriting the ndpp::List to do LIFO / FILO.
    Reference: https://en.cppreference.com/cpp/container/stack
*/
class Stack : private List
{
public:
    Stack() = default;
    inline Stack(Stack &&src);
    inline Stack(const Stack &src);

    virtual ~Stack() = default;

    /* 
        Used for the specified situation.
        Stack dst;
        dst = func(...);
    */
    inline Stack& operator=(const Stack &src);

    // Clone itself to new Stack.
    inline Stack clone() const;

    // Returns the number of elements.
    inline const size_t size() const;

    // Checks whether the container adaptor is empty.
    inline const bool empty() const;

    // Pushes the given element value to the top of the stack (used for move constructor).
    inline void push(Generic &&object);

    // Pushes the given element value to the top of the stack (used for copy constructor).
    inline void push(const Generic &object);

    // Accesses the top element.
    inline Generic& top() const;

    // Removes the top element.
    inline void pop();
};

inline Stack::Stack(Stack &&src) : List(std::move(src))
{
}

inline Stack::Stack(const Stack &src) : List(src)
{
}

inline Stack& Stack::operator=(const Stack &src)
{
    List::operator=(src);
    return *this;
}

inline Stack Stack::clone() const
{
    Stack _dst;
    _dst.copy(*this);
    return _dst; 
}

inline const size_t Stack::size() const
{
    return List::size();
}

inline const bool Stack::empty() const
{
    return (List::size() > 0) ? false : true;
}

inline void Stack::push(Generic &&object)
{
    pushFront(std::move(object));
}

inline void Stack::push(const Generic &object)
{
    pushFront(object);
}

inline Generic& Stack::top() const
{
    auto &top_node = front();
    return top_node.data();
}

inline void Stack::pop()
{
    erase(&front());
}

}; // namespace ndpp