#pragma once

#include <iostream>


namespace ndpp
{

namespace ndpp_memory
{

// Get sizeof(T) * num.
template<typename T>
inline size_t sizeOf(const size_t num)
{
    return num * sizeof(T);
}

};

};