#pragma once

#include <cstddef>

#define OMP_LOOP_NUM 1e+6


inline size_t threadsReq(const size_t loop_total)
{
    const size_t threads = 2 * int(loop_total / OMP_LOOP_NUM);
    return (threads > 0) ? threads : 1;
}