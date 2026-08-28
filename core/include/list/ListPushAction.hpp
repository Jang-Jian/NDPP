#pragma once

namespace ndpp
{

namespace ndpp_action
{

// Node push action.
enum class ListPushAction;


constexpr const char* _FrontStr = "front";
constexpr const char* _BackStr  = "back";

enum class ListPushAction
{
    // Pushing to front of specific node.
    Front = 0,

    // Pushing to back of specific node.
    Back  = 1
};

}; // namespace ndpp::ndpp_action

}; // namespace ndpp