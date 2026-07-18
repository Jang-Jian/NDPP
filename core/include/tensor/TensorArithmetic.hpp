#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

#include <include/base/Scalar.hpp>


namespace ndpp
{

// P.S Tensor is forward-declaring.
class Tensor;


namespace ndpp_arithmetic
{

/* 
    Arithmetic: Arithmetic method (C = A {operator} B).
    - Tensor operator+(A, B).
    - Tensor operator-(A, B).
    - Tensor operator*(A, B).
    - Tensor operator/(A, B).
*/
enum class Arithmetic;


// arithmeticForward: Arithmetic operation with forward which used for Tensor {+, -, * or /} Tensor.
// P.S Pay attention.
// - A & C has same scalar & device type, B would be independent.
// - A & C can use same data to calcuation.
void arithmeticForward(const Tensor &a, const Tensor &b, Tensor &c,
                       const Arithmetic arith_type, const std::string &operator_name); 


// arithmeticForward: Arithmetic operation with forward which used for Tensor {+, -, * or /} Scalar.
// P.S Pay attention.
// - A & C has same scalar & device type, B would be independent.
// - A & C can use same data to calcuation.
void arithmeticForward(const Tensor &a, const Scalar &b, Tensor &c,
                       const Arithmetic arith_type, const std::string &operator_name);


// arithmeticForward: Arithmetic operation with forward which used for Scalar {+, -, * or /} Tensor.
// P.S 
// - A & C has same scalar type. 
// - B & C has same device type.               
void arithmeticForward(const Scalar &a, const Tensor &b, Tensor &c,
                       const Arithmetic arith_type, const std::string &operator_name);


enum class Arithmetic
{
    // operator+
    Add      = 0,

    // operator-
    Subtract = 1,

    // opeator*.
    Multiply = 2,

    // opeator/.
    Division = 3
};


}; // namespace ndpp_arithmetic

}; // namespace ndpp