#pragma once

#include <include/list/List.hpp>


namespace ndpp
{

// Get specifc ListIterator via steps.
// P.S Used for ndpp::List, and it will replace originial ListIterator.
void advance(ListIterator &src, const std::ptrdiff_t steps);

// Get specifc ListIterator via steps.
// P.S Used for ndpp::List, and it will return a new ListIterator.
inline ListIterator next(ListIterator src, const std::ptrdiff_t steps)
{
    ndpp::advance(src, steps);
    return src;
}

} // namespace ndpp