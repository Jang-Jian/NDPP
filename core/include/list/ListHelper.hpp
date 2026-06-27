#pragma once

#include <include/list/List.hpp>


namespace ndpp
{

// Get specifc ListIterator via steps.
// P.S Used for ndpp::List.
ListIterator next(ListIterator src, const size_t steps);

} // namespace ndpp