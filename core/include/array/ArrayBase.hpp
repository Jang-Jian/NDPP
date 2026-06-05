#pragma once

#include <iostream>

#include <include/base/DeviceProc.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_array_base
{
 

/*
    ArrayDevice: Used for ndpp::Array, which processes memory processing.

    P.S The default allocation ArrayDevice<T>() has these the settings in below.
    _data = nullptr.
    _size = 0.
    _dtype = ndpp_memory::DeviceType::Host.
    _dstatus = ndpp_memory::DeviceStatus::Allocation.
*/
template<typename T>
class ArrayDevice
{
public:
    inline ArrayDevice();
    virtual inline ~ArrayDevice();

protected:
    inline void DeviceAlloc(const size_t size, const ndpp_memory::DeviceType device_type, 
                            const string &file_name, const string &method_name);

    inline void DeviceReset();

    inline void DeviceDeAlloc(const string &file_name, const string &method_name);

    inline void DeviceRefer(T *data, const size_t size, const ndpp_memory::DeviceType dtype,
                            const string &file_name, const string &method_name);

    inline void DeviceCopy(const T *src, const ndpp_memory::DeviceType src_device_type, 
                           const ndpp_memory::DeviceType dst_device_type, const size_t size, 
                           const string &file_name, const string &method_name);

    inline void DeviceCopyTo(T *dst, const ndpp_memory::DeviceType dst_device_type,
                             const string &file_name, const string &method_name);

    inline T* DevicePtr() const;
    inline size_t DeviceSize() const;
    inline ndpp_memory::DeviceType DeviceTy() const;
    inline ndpp_memory::DeviceStatus DeviceSt() const;

private:
    T *_data = nullptr;
    size_t _size = 0;

    ndpp_memory::DeviceType _dtype = ndpp_memory::DeviceType::Host;
    ndpp_memory::DeviceStatus _dstatus = ndpp_memory::DeviceStatus::Allocation;
};

template<typename T>
inline ArrayDevice<T>::ArrayDevice()
{
    DeviceReset();
}

template<typename T>
inline ArrayDevice<T>::~ArrayDevice()
{
    DeviceDeAlloc("ArrayBase.hpp", "ArrayDevice<T>::~ArrayDevice()");
}

template<typename T>
inline void ArrayDevice<T>::DeviceAlloc(const size_t size, const ndpp_memory::DeviceType device_type,
                                        const string &file_name, const string &method_name)
{
    DeviceDeAlloc(file_name, method_name);

    if (size > 0)
    {   
        this->_data = ndpp_memory::mixMemoryAlloc<T>(size, device_type, 
                                                     file_name, method_name);

        this->_size = (this->_data) ? size : 0;
        this->_dtype = device_type;
        this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
    }
}

template<typename T>
inline void ArrayDevice<T>::DeviceReset()
{
    this->_size = 0;
    this->_data = nullptr;
    this->_dtype = ndpp_memory::DeviceType::Host;
    this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
}

template<typename T>
inline void ArrayDevice<T>::DeviceDeAlloc(const string &file_name, const string &method_name)
{
    if (this->_data && this->_size > 0 && 
        this->_dstatus == ndpp_memory::DeviceStatus::Allocation)
    {
        ndpp_memory::mixMemoryDeAlloc(this->_data, this->_dtype, 
                                      file_name, method_name);
    }

    // Reset all settings.
    DeviceReset();
}

template<typename T>
inline void ArrayDevice<T>::DeviceRefer(T *data, const size_t size, const ndpp_memory::DeviceType dtype,
                                        const string &file_name, const string &method_name)
{
    DeviceDeAlloc(file_name, method_name);

    this->_data = data;
    this->_size = size;
    this->_dtype = dtype;
    this->_dstatus = ndpp_memory::DeviceStatus::Reference;
}

template<typename T>
inline void ArrayDevice<T>::DeviceCopy(const T *src, const ndpp_memory::DeviceType src_device_type, 
                                       const ndpp_memory::DeviceType dst_device_type, const size_t size, 
                                       const string &file_name, const string &method_name)
{
    DeviceAlloc(size, dst_device_type, file_name, method_name);
    
    ndpp_memory::mixMemoryCopy(src, src_device_type,
                               this->_data, this->_dtype, this->_size,
                               file_name, method_name);
}

template<typename T>
inline void ArrayDevice<T>::DeviceCopyTo(T *dst, const ndpp_memory::DeviceType dst_device_type,
                                         const string &file_name, const string &method_name)
{
    ndpp_memory::mixMemoryCopy(this->_data, this->_dtype,
                               dst, dst_device_type, this->_size,
                               file_name, method_name);
}

template<typename T>
inline T* ArrayDevice<T>::DevicePtr() const
{
    return this->_data;
}

template<typename T>
inline size_t ArrayDevice<T>::DeviceSize() const
{
    return this->_size;
}

template<typename T>
inline ndpp_memory::DeviceType ArrayDevice<T>::DeviceTy() const
{
    return this->_dtype;
}

template<typename T>
inline ndpp_memory::DeviceStatus ArrayDevice<T>::DeviceSt() const
{
    return this->_dstatus;
}

    
}; // namespace ndpp::array_base

}; // namespace ndpp