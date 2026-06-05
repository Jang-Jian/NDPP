#pragma once


namespace ndpp
{

namespace ndpp_memory
{

// DeviceType: Device(Memory) type.   
enum class DeviceType;

}; // namespace ndpp::ndpp_memory

namespace ndpp_memory
{


enum class DeviceType
{
    // Host: Host(Paged) memory allocation.
    Host        = 0,

#ifdef CUDA
    // CudaDevice: Cuda device memory allocation.
    CudaDevice  = 1,

    // CudaPinned: Cuda pinned(paged-Locked) memory allocation.
    CudaPinned  = 2,

    // CudaPinned: Cuda pinned(paged-Locked) memory allocation.
    CudaUnified = 3,

    // CudaZeroCpy: Cuda zero-copy memory allocation.
    CudaZeroCpy = 4
#endif
};


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp