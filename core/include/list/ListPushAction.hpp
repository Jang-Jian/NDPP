#pragma once

namespace ndpp
{

namespace ndpp_action
{

// Node push action.
enum class ListPushAction;

enum class ListPushAction
{
    // Pushing to front of specific node.
    Front = 0,

    // Pushing to back of specific node.
    Back  = 1
};

}; // namespace ndpp::ndpp_action

}; // namespace ndpp