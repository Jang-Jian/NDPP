#pragma once

#include <iterator>


namespace ndpp
{

namespace ndpp_python
{


/*
    PyListIterator: The basic iterator which is used for Python-C++ object iteration.
    P.S Please inherit this object and override these virtual operators: 
        1. PyBasicIterator::operator*().
        1. PyBasicIterator::operator++().
        2. PyBasicIterator::operator--().
        2. PyBasicIterator::operator--(int).
*/
template<class CcData, class PyData, class DerivedIterator>
class PyBasicIterator
{
public:
    using PyIteratorCategory = std::bidirectional_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using CcDataType        = CcData;
    using PyDataType        = PyData;
    using CcDataPointer     = CcData*;
    using PyDataPointer     = PyData*;
    using CcDataReference   = CcData&;
    using PyDataReference   = PyData&;

    inline explicit PyBasicIterator(CcDataPointer t_ptr) : _t_ptr(t_ptr) {}

    virtual PyDataType operator*() const = 0;

    virtual DerivedIterator& operator++() = 0;
    virtual DerivedIterator& operator--() = 0;

    virtual DerivedIterator operator++(int) = 0;

    inline bool operator==(const DerivedIterator &b) const { return this->_t_ptr == b._t_ptr; }
    inline bool operator!=(const DerivedIterator &b) const { return this->_t_ptr != b._t_ptr; }

    // Used for Python __iter__.
    virtual DerivedIterator& pyiter();

    // Used for Python __next__.
    virtual PyDataType pynext() = 0;

protected:
    CcDataPointer _t_ptr;
};

}; // namespace ndpp::ndpp_python

}; // namespace ndpp