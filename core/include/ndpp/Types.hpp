#pragma once

#include <include/base/ScalarType.hpp>
#include <include/base/DeviceType.hpp>


namespace ndpp
{

constexpr auto Host = ndpp_memory::DeviceType::Host;
#ifdef CUDA
constexpr auto CudaDevice = ndpp_memory::DeviceType::CudaDevice;
constexpr auto CudaPinned = ndpp_memory::DeviceType::CudaPinned;
constexpr auto CudaUnified = ndpp_memory::DeviceType::CudaUnified;
constexpr auto CudaZeroCpy = ndpp_memory::DeviceType::CudaZeroCpy;
#endif


constexpr auto UInt8 = ndpp_memory::ScalarType::UInt8;
constexpr auto UInt16 = ndpp_memory::ScalarType::UInt16;
constexpr auto UInt32 = ndpp_memory::ScalarType::UInt32;
constexpr auto UInt64 = ndpp_memory::ScalarType::UInt32;
constexpr auto Int8 = ndpp_memory::ScalarType::Int8;
constexpr auto Int16 = ndpp_memory::ScalarType::Int16;
constexpr auto Int32 = ndpp_memory::ScalarType::Int32;
constexpr auto Int64 = ndpp_memory::ScalarType::Int64;
#ifdef HALF
constexpr auto Float16 = ndpp_memory::ScalarType::Float16;
#endif
constexpr auto Float32 = ndpp_memory::ScalarType::Float32;
constexpr auto Float64 = ndpp_memory::ScalarType::Float64;
constexpr auto Bool = ndpp_memory::ScalarType::Bool;    

};