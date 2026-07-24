#pragma once

#include <omp.h>
#include <cstddef>


namespace ndpp
{

// simd_length: Used for openmp simdlen().
constexpr size_t simd_length = 4;


// threadsReq: number of threads request for openmp threading.
inline size_t threadsReq(const size_t loop_total)
{
    constexpr float parallel_std_size = 1e+6f;
    constexpr float ratio = 2.0f;
    constexpr size_t no_threads = 0;
    constexpr size_t default_threads = 1;

    const size_t threads = ratio * size_t(loop_total / parallel_std_size);
    //const size_t avail_threads = static_cast<size_t>(omp_get_num_threads());

    return (threads > no_threads) ? threads : default_threads;
}

}; // namespace ndpp