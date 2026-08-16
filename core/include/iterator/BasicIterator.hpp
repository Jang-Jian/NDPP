#pragma once

#include <iterator>

#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_iterator
{


/*
    BasicIterator: The basic iterator which is used for object iteration.
    P.S Please inherit this object and implement 2 virtual operators: operator++() & operator--().
*/
template<class T>
class BasicIterator 
{
public:
    // C++ STL Iterator traits (Required for full compatibility)
    using iterator_category = std::forward_iterator_tag;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T&;

    // Constructor
    inline explicit BasicIterator(pointer t_ptr) : _t_ptr(t_ptr) {}

    // 1. Dereference operator
    inline reference operator*() 
    { 
        if (!_t_ptr)
        {
            ndpp_log::logger("BasicIterator.hpp", "BasicIterator<T>::operator*()", ndpp_log::RuntimeType::Error,
                             "Self pointer is nullptr.", true);
            exit(EXIT_FAILURE);
        }
        return *_t_ptr; 
    }
    //pointer operator->() const { return _t_ptr; }

    // 2. Prefix increment operator
    virtual BasicIterator& operator++() = 0;

    virtual BasicIterator& operator--() = 0;

    // Postfix increment operator (Optional, but good practice)
    /*BasicIterator operator++(int) {
        BasicIterator tmp = *this;
        ++(*this);
        return tmp;
    }*/

    // 3. Comparison operators
    friend bool operator==(const BasicIterator& a, const BasicIterator& b) { return a._t_ptr == b._t_ptr; }
    friend bool operator!=(const BasicIterator& a, const BasicIterator& b) { return a._t_ptr != b._t_ptr; }

protected:
    pointer _t_ptr;
};

}; // namespace ndpp::ndpp_iterator

}; // namespace ndpp