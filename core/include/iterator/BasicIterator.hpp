#pragma once

#include <iterator>

#include <include/logging/Logging.hpp>


namespace ndpp
{

namespace ndpp_iterator
{


/*
    BasicIterator: The basic iterator which is used for C++ object iteration.
    P.S Please inherit this object and override these virtual operators: 
        1. BasicIterator::operator++().
        2. BasicIterator::operator--().
        2. BasicIterator::operator--(int).
*/
template<class Data, class DerivedIterator>
class BasicIterator 
{
public:
    // C++ STL Iterator traits (Required for full compatibility)
    using IteratorCategory = std::forward_iterator_tag;
    using DataType         = Data;
    using DataPointer      = Data*;
    using DataReference    = Data&;

    // Constructor
    inline explicit BasicIterator(DataPointer t_ptr) : _t_ptr(t_ptr) {}

    // 1. Dereference operator.
    inline DataReference operator*() { return *_t_ptr; }
    inline DataPointer operator->() const { return _t_ptr; }

    // 2. Prefix increment operator (++i).
    virtual DerivedIterator& operator++() = 0;
    virtual DerivedIterator& operator--() = 0;

    // 3. Postfix increment operator (i++).
    virtual DerivedIterator operator++(int) = 0;

    // 4. Comparison operators.
    inline bool operator==(const DerivedIterator &b) const { return this->_t_ptr == b._t_ptr; }
    inline bool operator!=(const DerivedIterator &b) const { return this->_t_ptr != b._t_ptr; }

protected:
    DataPointer _t_ptr;
};

}; // namespace ndpp::ndpp_iterator

}; // namespace ndpp