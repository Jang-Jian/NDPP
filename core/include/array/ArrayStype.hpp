#pragma once

#include <include/array/Array.hpp>


namespace ndpp
{


/* 
    Scalar-Array.
    Array<size_t>: SizeTArray.
*/
typedef Array<size_t> SizeTArray;


// True if two tensors have the same size and elements, False otherwise.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.equal.html
bool same(const SizeTArray &src1, const SizeTArray &src2);


}; // namespace ndpp