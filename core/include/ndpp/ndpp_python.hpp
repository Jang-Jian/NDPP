#pragma once


#include <include/ndpp/ndpp.hpp>

#include <include/python/PyScalar.hpp>
#include <include/python/PyHelper.hpp>
#include <include/python/PyTensor.hpp>
#include <include/python/PyTensorProc.hpp>
#include <include/python/PyExtension.hpp>


namespace ndpp
{

namespace ndpp_python
{

constexpr const char* _NoneStr      = "none";
constexpr const char* _ReversedStr  = "reversed";

constexpr const auto None     = PyOpPyTensor::None;
constexpr const auto Reversed = PyOpPyTensor::Reversed;

}; // namespace ndpp::ndpp_python

}; // namespace ndpp