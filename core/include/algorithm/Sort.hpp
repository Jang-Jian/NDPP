#pragma once

#include <include/tensor/Tensor.hpp>


namespace ndpp
{

// Sorts the elements of the src tensor along a given dimension in ascending order by value.
// P.S 
//  - The stable only supports true only, it will be supported false in the next update.
//  - Conducting via CPU only.
// Reference: https://docs.pytorch.org/docs/2.12/generated/torch.sort.html
Tensor sort(const Tensor &src, int64_t dim = -1, const bool descending = false, const bool stable = true);

}; // namespace ndpp