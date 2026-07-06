#ifndef __CudaProc_hpp__
#define __CudaProc_hpp__

#include <cmath>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#ifdef HALF
#include <cuda_fp16.h>
#endif
//#include <cublas_v2.h>
//

#include <include/logging/Logging.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_memory
{

namespace ndpp_cuda
{


// CudaDeviceType: CUDA device(memory) type.
enum class CudaDeviceType
{
    // CudaDevice: Cuda device memory allocation.
    CudaDevice  = 0,

    // CudaPinned: Cuda pinned(paged-Locked) memory allocation.
    CudaPinned  = 1,

    // CudaZeroCpy: Cuda zero-copy memory allocation.
    CudaUnified = 2,

    // CudaDevice: Cuda device memory allocation.
    CudaZeroCpy = 3
};


// Check wether to has error after cuda exection.
inline bool cudaErrorChecker(const cudaError_t status, const string &file_name, const string &method_name)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    //bool show_info = false;

    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        ndpp_log::logger(file_name, method_name, ndpp_log::RuntimeType::ERROR, 
                         "In ndpp::ndpp_memory::ndpp_cuda::cudaErrorChecker(), " + string(s) + ".", true);
        return false;
    }

    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        ndpp_log::logger(file_name, method_name, ndpp_log::RuntimeType::ERROR, 
                         "In ndpp::ndpp_memory::ndpp_cuda::cudaErrorChecker(), " + string(s) + ".", true);
        return false;
	}

    return true;
}


// Memory allocation with different cuda's memory types.
template<typename dtype>
inline dtype* cudaMemoryAlloc(const size_t size, const CudaDeviceType cudevice_type, 
                              const string &file_name, const string &method_name)
{
    if (size == 0)
    {
        return nullptr;
    }

    bool status = false;
    dtype *dst_ptr = nullptr;
    
    switch (cudevice_type)
    {
    case CudaDeviceType::CudaDevice:
        status = cudaErrorChecker(cudaMalloc((void**)&dst_ptr, sizeof(dtype) * size), file_name, method_name);
        break;
    case CudaDeviceType::CudaPinned:
        status = cudaErrorChecker(cudaMallocHost((void**)&dst_ptr, sizeof(dtype) * size), file_name, method_name);
        break;
    case CudaDeviceType::CudaUnified:
        status = cudaErrorChecker(cudaMallocManaged((void**)&dst_ptr, sizeof(dtype) * size), file_name, method_name);
        break;
    case CudaDeviceType::CudaZeroCpy:
        status = cudaErrorChecker(cudaHostAlloc((void**)&dst_ptr, sizeof(dtype) * size, cudaHostAllocMapped), file_name, method_name);
        break;
    }

    if (!status)
    {
        dst_ptr = nullptr;
    }
    
    return dst_ptr;
}


// Memory deallocation with different cuda's memory types.
inline void cudaMemoryDeAlloc(void *src_ptr, const CudaDeviceType cudevice_type, 
                              const string &file_name, const string &method_name)
{
    if (!src_ptr)
    {
        return;
    }

    switch (cudevice_type)
    {
    case CudaDeviceType::CudaDevice:
    case CudaDeviceType::CudaUnified:
        cudaErrorChecker(cudaFree(src_ptr), file_name, method_name);
        break;
    case CudaDeviceType::CudaPinned:
    case CudaDeviceType::CudaZeroCpy:
        cudaErrorChecker(cudaFreeHost(src_ptr), file_name, method_name);
        break;
    }
}


/* 
    Memory copy with different memory types.
    @para
    - kind: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost.
*/
inline void cudaMemoryCopy(const void *src_ptr, void *dst_ptr, const cudaMemcpyKind kind, const size_t size,
                           const string &file_name, const string &method_name)
{
    if (!src_ptr)
    {
        return;
    }
    if (!dst_ptr)
    {
        return;
    }

    cudaErrorChecker(cudaMemcpy(dst_ptr, src_ptr, size, kind), file_name, method_name);
}


/*
    Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
    Used for ndpp::ndpp_memory::CudaDeviceType::CudaZeroCpy
    reference: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc00502b44e5f1bdc0b424487ebb08db0
*/
inline void* cudaHostGetGpuPointer(const void *src_ptr, const string &file_name, const string &method_name)
{
    if (!src_ptr)
    {
        return nullptr;
    }

    void *dst_ptr = nullptr;
    cudaErrorChecker(cudaHostGetDevicePointer((void**)&dst_ptr, (void*)src_ptr, 0), file_name, method_name);

    return dst_ptr;
}


// Configure the cuda device id.
inline bool cudaExecDevice(const int device, const string &file_name, const string &method_name)
{
    return cudaErrorChecker(cudaSetDevice(device), file_name, method_name);
}


// Allocate cuda executed grid size.
// Reference: https://github.com/pjreddie/darknet/blob/master/src/cuda.c#L51
inline dim3 cudaExecGrid(const size_t all_threads_num, const size_t per_block_threads_num)
{
    size_t k = (all_threads_num - 1) / per_block_threads_num + 1;
    size_t x = k;
    size_t y = 1;
    
    if (x > 65535)
    {
        x = static_cast<size_t>(ceil(sqrt(k)));
        y = static_cast<size_t>((all_threads_num - 1) / (x * per_block_threads_num) + 1);
    }

    return dim3(x, y, 1);
}


}; // namespace ndpp::ndpp_memory::ndpp_cuda

}; // namespace ndpp::ndpp_memory

}; // namespace ndpp


#endif