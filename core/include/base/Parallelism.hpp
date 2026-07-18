#pragma once

#include <cstddef>


namespace ndpp
{

// threadsReq: number of threads request for openmp threading.
inline size_t threadsReq(const size_t loop_total)
{
    constexpr float parallel_std_size = 1e+6f;
    const size_t threads = 2 * int(loop_total / parallel_std_size);
    return (threads > 0) ? threads : 1;
}

}; // namespace ndpp