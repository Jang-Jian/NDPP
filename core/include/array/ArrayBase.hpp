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
template<class T>
class ArrayDevice
{
public:
    inline ArrayDevice();
    inline ArrayDevice(ArrayDevice<T> &&src);
    inline ArrayDevice(const ArrayDevice<T> &src);

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

template<class T>
inline ArrayDevice<T>::ArrayDevice()
{
    DeviceReset();
}

template<class T>
inline ArrayDevice<T>::ArrayDevice(ArrayDevice<T> &&src) : _data(src._data), _size(src._size), 
                                                           _dtype(src._dtype), _dstatus(src._dstatus)
{
    src._data = nullptr;
    src._size = 0;
    src._dtype = ndpp_memory::DeviceType::Host; 
    src._dstatus = ndpp_memory::DeviceStatus::Allocation;
}

template<class T>
inline ArrayDevice<T>::ArrayDevice(const ArrayDevice<T> &src)
{
    if (this == &src)
    {
        return;
    }

    switch (src._dstatus)
    {
    case ndpp_memory::DeviceStatus::Allocation:
        DeviceCopy(src._data, src._dtype, src._dtype, src._size, 
                   "ArrayBase.hpp", "ArrayDevice<T>::ArrayDevice(const ArrayDevice<T>&)");
        break;
    case ndpp_memory::DeviceStatus::Reference:
        DeviceRefer(src._data, src._size, src._dtype,
                    "ArrayBase.hpp", "ArrayDevice<T>::ArrayDevice(const ArrayDevice<T>&)");
        break;
    }
}

template<class T>
inline ArrayDevice<T>::~ArrayDevice()
{
    DeviceDeAlloc("ArrayBase.hpp", "ArrayDevice<T>::~ArrayDevice()");
}

template<class T>
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

template<class T>
inline void ArrayDevice<T>::DeviceReset()
{
    this->_size = 0;
    this->_data = nullptr;
    this->_dtype = ndpp_memory::DeviceType::Host;
    this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
}

template<class T>
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

template<class T>
inline void ArrayDevice<T>::DeviceRefer(T *data, const size_t size, const ndpp_memory::DeviceType dtype,
                                        const string &file_name, const string &method_name)
{
    if (this->_data == data)
    {
        return;
    }

    DeviceDeAlloc(file_name, method_name);

    this->_data = data;
    this->_size = size;
    this->_dtype = dtype;
    this->_dstatus = ndpp_memory::DeviceStatus::Reference;
}

template<class T>
inline void ArrayDevice<T>::DeviceCopy(const T *src, const ndpp_memory::DeviceType src_device_type, 
                                       const ndpp_memory::DeviceType dst_device_type, const size_t size, 
                                       const string &file_name, const string &method_name)
{
    if (this->_data == src)
    {
        return;
    }

    DeviceAlloc(size, dst_device_type, file_name, method_name);
    
    ndpp_memory::mixMemoryCopy(src, src_device_type,
                               this->_data, this->_dtype, this->_size,
                               file_name, method_name);
}

template<class T>
inline void ArrayDevice<T>::DeviceCopyTo(T *dst, const ndpp_memory::DeviceType dst_device_type,
                                         const string &file_name, const string &method_name)
{
    ndpp_memory::mixMemoryCopy(this->_data, this->_dtype,
                               dst, dst_device_type, this->_size,
                               file_name, method_name);
}

template<class T>
inline T* ArrayDevice<T>::DevicePtr() const
{
    return this->_data;
}

template<class T>
inline size_t ArrayDevice<T>::DeviceSize() const
{
    return this->_size;
}

template<class T>
inline ndpp_memory::DeviceType ArrayDevice<T>::DeviceTy() const
{
    return this->_dtype;
}

template<class T>
inline ndpp_memory::DeviceStatus ArrayDevice<T>::DeviceSt() const
{
    return this->_dstatus;
}

    
}; // namespace ndpp::array_base

}; // namespace ndpp