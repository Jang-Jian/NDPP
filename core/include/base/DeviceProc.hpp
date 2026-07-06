#pragma once

#include <iostream>

#include <include/base/DeviceType.hpp>
#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif

using namespace std;


namespace ndpp
{

namespace ndpp_memory
{

// DeviceStatus: Device Status.
enum class DeviceStatus;

// mixMemoryAlloc: Dynamic Memory Allocation which mixes host & cuda.
template<typename dtype>
inline dtype* mixMemoryAlloc(const size_t size, const DeviceType device_type,
                             const string &file_name, const string &method_name);

// mixMemoryDeAlloc: Dynamic Memory Deallocation which mixes host & cuda.
template<typename dtype>
inline void mixMemoryDeAlloc(dtype *src, const DeviceType device_type,
                             const string &file_name, const string &method_name);

// mixMemoryDeAlloc: Dynamic Memory Deallocation which mixes host & cuda.
template<typename dtype>
inline void mixMemoryCopy(const dtype *src, const DeviceType src_device_type,
                          dtype *dst, const DeviceType dst_device_type, const size_t size,
                          const string &file_name, const string &method_name);

}; // namespace ndpp::ndpp_memory


namespace ndpp_memory
{



enum class DeviceStatus
{
    // Allocation: The memory has allocated.
    Allocation = 1,

    // Reference:  The memory has referenced.
    Reference  = 2  
};


template<typename dtype>
inline dtype* mixMemoryAlloc(const size_t size, const DeviceType device_type,
                             const string &file_name, const string &method_name)
{
    if (size == 0)
    {
        return nullptr;
    }

    dtype *dst = nullptr;

    switch (device_type)
    {
        case DeviceType::Host:
            dst = new dtype[size]();
            break;
    #ifdef CUDA
        case DeviceType::CudaDevice:
            dst = ndpp_cuda::cudaMemoryAlloc<dtype>(size, ndpp_cuda::CudaDeviceType::CudaDevice, 
                                                    file_name, method_name);
            break;
        case DeviceType::CudaPinned:
            dst = ndpp_cuda::cudaMemoryAlloc<dtype>(size, ndpp_cuda::CudaDeviceType::CudaPinned, 
                                                    file_name, method_name);
            break;
        case DeviceType::CudaUnified:
            dst = ndpp_cuda::cudaMemoryAlloc<dtype>(size, ndpp_cuda::CudaDeviceType::CudaUnified, 
                                                    file_name, method_name);
            break;
        case DeviceType::CudaZeroCpy:
            dst = ndpp_cuda::cudaMemoryAlloc<dtype>(size, ndpp_cuda::CudaDeviceType::CudaZeroCpy, 
                                                    file_name, method_name);
            break;
    #endif
    }

    return dst;
}


template<typename dtype>
inline void mixMemoryDeAlloc(dtype *src, const DeviceType device_type,
                             const string &file_name, const string &method_name)
{
    if (!src)
    {
        return;
    }

    switch (device_type)
    {
        case DeviceType::Host:
            delete[] src;
            break;
    #ifdef CUDA
        case DeviceType::CudaDevice:
            ndpp_cuda::cudaMemoryDeAlloc(src, ndpp_cuda::CudaDeviceType::CudaDevice, 
                                        file_name, method_name);
            break;
        case DeviceType::CudaPinned:
            ndpp_cuda::cudaMemoryDeAlloc(src, ndpp_cuda::CudaDeviceType::CudaPinned, 
                                        file_name, method_name);
            break;
        case DeviceType::CudaUnified:
            ndpp_cuda::cudaMemoryDeAlloc(src, ndpp_cuda::CudaDeviceType::CudaUnified, 
                                        file_name, method_name);
            break;
        case DeviceType::CudaZeroCpy:
            ndpp_cuda::cudaMemoryDeAlloc(src, ndpp_cuda::CudaDeviceType::CudaZeroCpy, 
                                        file_name, method_name);
            break;
    #endif
    }
}


template<typename dtype>
inline void mixMemoryCopy(const dtype *src, const DeviceType src_device_type,
                          dtype *dst, const DeviceType dst_device_type, const size_t size,
                          const string &file_name, const string &method_name)
{
    if (!src || !dst || size == 0)
    {
        return;
    }

    switch (src_device_type)
    {
        case DeviceType::Host:
    #ifdef CUDA
        case DeviceType::CudaPinned:
        case DeviceType::CudaUnified:
    #endif
            switch (dst_device_type)
            {
                case DeviceType::Host:
            #ifdef CUDA
                case DeviceType::CudaPinned:
                case DeviceType::CudaUnified:
                case DeviceType::CudaZeroCpy:
            #endif
                    std::copy(src, src + size, dst);
                    break;

            #ifdef CUDA
                case DeviceType::CudaDevice:    
                    ndpp_cuda::cudaMemoryCopy(src, dst, cudaMemcpyHostToDevice, size * sizeof(dtype),
                                              file_name, method_name);
                    break;
            #endif
            }
            break;

    #ifdef CUDA
        case DeviceType::CudaDevice:
            switch (dst_device_type)
            {
                case DeviceType::Host:
                case DeviceType::CudaPinned:
                case DeviceType::CudaUnified:
                    ndpp_cuda::cudaMemoryCopy(src, dst, cudaMemcpyDeviceToHost, size * sizeof(dtype),
                                            file_name, method_name);
                    break;
                case DeviceType::CudaDevice:
                    ndpp_cuda::cudaMemoryCopy(src, dst, cudaMemcpyDeviceToDevice, size * sizeof(dtype),
                                            file_name, method_name);
                    break;
                case DeviceType::CudaZeroCpy:
                    ndpp_cuda::cudaMemoryCopy(src, ndpp_cuda::cudaHostGetGpuPointer(dst, file_name, method_name), 
                                              cudaMemcpyDeviceToDevice, size * sizeof(dtype), file_name, method_name);
                    break;
            }
            break;

        case DeviceType::CudaZeroCpy:
            switch (dst_device_type)
            {
                case DeviceType::Host:
                case DeviceType::CudaPinned:
                case DeviceType::CudaUnified:
                case DeviceType::CudaZeroCpy:      
                    std::copy(src, src + size, dst);
                    break;

                case DeviceType::CudaDevice:
                    ndpp_cuda::cudaMemoryCopy(ndpp_cuda::cudaHostGetGpuPointer(src, file_name, method_name), dst, 
                                              cudaMemcpyDeviceToDevice, size * sizeof(dtype), file_name, method_name);
                    break;
            }
            break;
    #endif
    }
}


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp