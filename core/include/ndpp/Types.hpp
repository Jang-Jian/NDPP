#pragma once

#include <include/base/ScalarType.hpp>
#include <include/base/DeviceType.hpp>
#include <include/base/DataProc.hpp>


namespace ndpp
{

constexpr const auto Allocation = ndpp_memory::DeviceStatus::Allocation;
constexpr const auto Reference = ndpp_memory::DeviceStatus::Reference;


constexpr const auto Host        = ndpp_memory::DeviceType::Host;
#ifdef CUDA
constexpr const auto CudaDevice  = ndpp_memory::DeviceType::CudaDevice;
constexpr const auto CudaPinned  = ndpp_memory::DeviceType::CudaPinned;
constexpr const auto CudaUnified = ndpp_memory::DeviceType::CudaUnified;
constexpr const auto CudaZeroCpy = ndpp_memory::DeviceType::CudaZeroCpy;
#endif


constexpr const auto UInt8   = ndpp_memory::ScalarType::UInt8;
constexpr const auto UInt16  = ndpp_memory::ScalarType::UInt16;
constexpr const auto UInt32  = ndpp_memory::ScalarType::UInt32;
constexpr const auto UInt64  = ndpp_memory::ScalarType::UInt32;
constexpr const auto Int8    = ndpp_memory::ScalarType::Int8;
constexpr const auto Int16   = ndpp_memory::ScalarType::Int16;
constexpr const auto Int32   = ndpp_memory::ScalarType::Int32;
constexpr const auto Int64   = ndpp_memory::ScalarType::Int64;
#ifdef HALF
constexpr const auto Float16 = ndpp_memory::ScalarType::Float16;
#endif
constexpr const auto Float32 = ndpp_memory::ScalarType::Float32;
constexpr const auto Float64 = ndpp_memory::ScalarType::Float64;
constexpr const auto Bool    = ndpp_memory::ScalarType::Bool;    


constexpr const auto Info  = ndpp_log::RuntimeType::Info;
constexpr const auto Error = ndpp_log::RuntimeType::Error;
constexpr const auto Warn  = ndpp_log::RuntimeType::Warn;
constexpr const auto Debug = ndpp_log::RuntimeType::Debug;
constexpr const auto End   = ndpp_log::RuntimeType::End;


}; // namespace ndpp