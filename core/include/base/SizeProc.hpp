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
    constexpr size_t data_size = sizeof(T);
    return num * data_size;
}

};

};