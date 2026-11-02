#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

using namespace std;


namespace ndpp
{

namespace ndpp_tensor_node
{

// ============================================================
// TensorNode
// Computational graph + memory lifetime management
// ============================================================
struct TensorNode 
{
    vector<shared_ptr<TensorNode>> parents;
    function<void()> backward_fn;

    // Own the allocations.
    // Tensor::data and Tensor::grad are non-owning aliases.
    shared_ptr<float[]> data_owner;
    shared_ptr<float[]> grad_owner;
};


};

};