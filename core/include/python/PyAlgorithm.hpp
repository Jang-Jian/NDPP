#pragma once

#include <include/algorithm/Sort.hpp>
#include <include/python/PyTensor.hpp>


namespace ndpp
{

namespace ndpp_python
{

PyTensor pySort(const PyTensor &src, const int64_t dim, const bool descending, const bool stable)
{
    Tensor _dst = sort(src, dim, descending, stable);
    return PyTensor(std::move(_dst));
}

}; // namespace ndpp::ndpp_python

}; // namespace ndpp