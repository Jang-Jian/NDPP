#pragma once


namespace ndpp
{

// The configuration used for 64-bit or 32-bit  environment.
#if defined(_WIN64) || defined(__x86_64__) || defined(__ppc64__) || defined(__aarch64__)
// Integer: 64-bit integer.
using Integer = int64_t;
#else
// Integer: 32-bit integer.
using Integer = int32_t;
#endif

}; // namespace ndpp