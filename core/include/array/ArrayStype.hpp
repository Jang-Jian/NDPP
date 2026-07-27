#pragma once

#include <include/array/Array.hpp>
#include <include/base/BasicType.hpp>


namespace ndpp
{

// IntArray: Integer Array.
// P.S Integer will be compiled 32-bit or 64-bit via host environment.
using IntArray = Array<Integer>;


// True if two tensors have the same size and elements, False otherwise.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.equal.html
bool same(const IntArray &src1, const IntArray &src2);


}; // namespace ndpp