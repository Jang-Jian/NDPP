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
template<typename T>
inline T* mixMemoryAlloc(const size_t size, const DeviceType device_type,
                             const string &file_name, const string &method_name);

// mixMemoryDeAlloc: Dynamic Memory Deallocation which mixes host & cuda.
template<typename T>
inline void mixMemoryDeAlloc(T *src, const DeviceType device_type,
                             const string &file_name, const string &method_name);

// mixMemoryDeAlloc: Dynamic Memory Deallocation which mixes host & cuda.
template<typename T>
inline void mixMemoryCopy(const T *src, const DeviceType src_device_type,
                          T *dst, const DeviceType dst_device_type, const size_t size,
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


template<typename T>
inline T* mixMemoryAlloc(const size_t size, const DeviceType device_type,
                             const string &file_name, const string &method_name)
{
    if (size == 0)
    {
        return nullptr;
    }

    T *dst = nullptr;

    switch (device_type)
    {
        case DeviceType::Host:
            dst = new T[size]();
            break;
    #ifdef CUDA
        case DeviceType::CudaDevice:
            dst = ndpp_cuda::cudaMemoryAlloc<T>(size, ndpp_cuda::CudaDeviceType::CudaDevice, 
                                                file_name, method_name);
            break;
        case DeviceType::CudaPinned:
            dst = ndpp_cuda::cudaMemoryAlloc<T>(size, ndpp_cuda::CudaDeviceType::CudaPinned, 
                                                file_name, method_name);
            break;
        case DeviceType::CudaUnified:
            dst = ndpp_cuda::cudaMemoryAlloc<T>(size, ndpp_cuda::CudaDeviceType::CudaUnified, 
                                                file_name, method_name);
            break;
        case DeviceType::CudaZeroCpy:
            dst = ndpp_cuda::cudaMemoryAlloc<T>(size, ndpp_cuda::CudaDeviceType::CudaZeroCpy, 
                                                file_name, method_name);
            break;
    #endif
    }

    return dst;
}


template<typename T>
inline void mixMemoryDeAlloc(T *src, const DeviceType device_type,
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


template<typename T>
inline void mixMemoryCopy(const T *src, const DeviceType src_device_type,
                          T *dst, const DeviceType dst_device_type, const size_t size,
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
                    ndpp_cuda::cudaMemoryCopy<T>(src, dst, cudaMemcpyHostToDevice, size,
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
                    ndpp_cuda::cudaMemoryCopy<T>(src, dst, cudaMemcpyDeviceToHost, size,
                                            file_name, method_name);
                    break;
                case DeviceType::CudaDevice:
                    ndpp_cuda::cudaMemoryCopy<T>(src, dst, cudaMemcpyDeviceToDevice, size,
                                                 file_name, method_name);
                    break;
                case DeviceType::CudaZeroCpy:
                    ndpp_cuda::cudaMemoryCopy<T>(src, ndpp_cuda::cudaHostGetGpuPointer(dst, file_name, method_name), 
                                                 cudaMemcpyDeviceToDevice, size, file_name, method_name);
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
                    ndpp_cuda::cudaMemoryCopy<T>(ndpp_cuda::cudaHostGetGpuPointer(src, file_name, method_name), dst, 
                                                 cudaMemcpyDeviceToDevice, size, file_name, method_name);
                    break;
            }
            break;
    #endif
    }
}


}; // namespace ndpp::ndpp_memory

}; // namespace ndpp