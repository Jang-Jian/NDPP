#pragma once

#include <cstdint>
#include <initializer_list>

#include <include/array/ArrayBase.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

/*
    Array: A Container that encapsulates fixed size arrays.
    Reference: https://en.cppreference.com/cpp/container/array
*/
template<class T>
class Array : private ndpp_array_base::ArrayDevice<T>
{
public:
    Array() = default;
    virtual ~Array() = default;
    inline Array(const Array<T> &src);
    inline Array(const initializer_list<T> &src); // host memory only.
    inline Array(const size_t size, const ndpp_memory::DeviceType dtype);
    inline Array(T *data, const size_t size, const ndpp_memory::DeviceType dtype);
    

    // Access specified element without bounds checking.
    inline T& operator[](const size_t pos) const;

    /* 
        Used for the specified situation.
        Array<T> dst;
        dst = func(...);
    */
    inline Array<T>& operator=(const Array<T> &src);

    // Access specified element with bounds checking.
    inline T& at(const size_t pos);

    // Returns the number of elements.
    inline size_t size() const;
    
    // Direct access to the underlying contiguous storage.
    inline T* data() const;

    // Returns an iterator to the beginning.
    inline T* begin() const;

    // Returns an iterator to the end (begin() + size).
    inline T* end() const;

    // Clone itself to new Array<T> via ndpp_memory::DeviceType.
    inline Array<T> clone(const ndpp_memory::DeviceType dtype);

    // Get the ndpp_memory::DeviceType.
    inline ndpp_memory::DeviceType device() const; 

    // Get the ndpp_memory::DeviceStatus.
    inline ndpp_memory::DeviceStatus status() const;

    // Copy other Array<T> to itself via ndpp_memory::DeviceType.
    inline void copy(const Array<T> &src, const ndpp_memory::DeviceType dtype);

    // Reference the data from other via pointer.
    inline void refer(T *data, const size_t size, const ndpp_memory::DeviceType dtype);

    // Reference the data from other via ndpp::Array.
    inline void refer(const Array<T> &src);

    // Migrating all information outside, and self will be cleaned.
    // P.S Please carefully use this function, pointer 'dst' must be managed or destoryed after program-end.
    inline void migrateTo(T **dst, size_t &array_size, 
                          ndpp_memory::DeviceType &dtype,
                          ndpp_memory::DeviceStatus &dstatus);

    // Allocate the memory with different ndpp_memory::DeviceType.
    inline void allocate(const size_t size, const ndpp_memory::DeviceType dtype);

    // Deallocate the memory.
    inline void destory();
};

template<class T>
inline Array<T>::Array(const Array<T> &src)
{
    switch (src.status())
    {
    case ndpp_memory::DeviceStatus::Allocation:
        copy(src, src.device());
        break;
    case ndpp_memory::DeviceStatus::Reference:
        refer(src);
        break;
    }
}

template<class T>
inline Array<T>::Array(const initializer_list<T> &src)
{
    allocate(src.size(), ndpp_memory::DeviceType::Host);
    T *_data = data();

    int index = -1;
    for (T val : src)
    {
        ++index;
        _data[index] = val;
    }
}

template<class T>
inline Array<T>::Array(const size_t size, const ndpp_memory::DeviceType dtype)
{
    allocate(size, dtype);
}

template<class T>
inline Array<T>::Array(T *data, const size_t size, const ndpp_memory::DeviceType dtype)
{
    refer(data, size, dtype);
}

template<class T>
inline T& Array<T>::operator[](const size_t pos) const
{
    return data()[pos];
}

template<class T>
inline Array<T>& Array<T>::operator=(const Array<T> &src)
{
    copy(src, src.device());
    return *this;
}

template<class T>
inline T& Array<T>::at(const size_t pos)
{
    size_t _size = size();

    if (_size == 0)
    {
        ndpp_log::logger("Array.hpp", "Array<T>::at()", ndpp_log::RuntimeType::ERROR, 
                         "The size of array is 0.", true);
        exit(EXIT_FAILURE); // terminating the process.
    }

    if (pos >= _size)
    {
        ndpp_log::logger("Array.hpp", "Array<T>::at()", ndpp_log::RuntimeType::ERROR, 
                         "Index(pos) is out of range.", true);
        exit(EXIT_FAILURE); // terminating the process.
    }

    return data()[pos];
}

template<class T>
inline size_t Array<T>::size() const
{
    return ndpp_array_base::ArrayDevice<T>::DeviceSize();
}

template<class T>
inline T* Array<T>::data() const
{
    return ndpp_array_base::ArrayDevice<T>::DevicePtr();
}

template<class T>
inline T* Array<T>::begin() const
{
    return ndpp_array_base::ArrayDevice<T>::DevicePtr();
}

template<class T>
inline T* Array<T>::end() const
{
    return ndpp_array_base::ArrayDevice<T>::DevicePtr() + 
           ndpp_array_base::ArrayDevice<T>::DeviceSize();
}

template<class T>
inline Array<T> Array<T>::clone(const ndpp_memory::DeviceType dtype)
{
    Array<T> _dst(size(), dtype);
    ndpp_array_base::ArrayDevice<T>::DeviceCopyTo(_dst.data(), dtype,
                                                  "Array.hpp", "Array<T>::clone()");
    return _dst;
}

template<class T>
inline ndpp_memory::DeviceType Array<T>::device() const
{
    return ndpp_array_base::ArrayDevice<T>::DeviceTy();
}

template<class T>
inline ndpp_memory::DeviceStatus Array<T>::status() const
{
    return ndpp_array_base::ArrayDevice<T>::DeviceSt();
}

template<class T>
inline void Array<T>::copy(const Array<T> &src, const ndpp_memory::DeviceType dtype)
{
    ndpp_array_base::ArrayDevice<T>::DeviceCopy(src.data(), src.device(), dtype, src.size(), 
                                                "Array.hpp", "Array<T>::copy()");
}

template<class T>
inline void Array<T>::refer(T *data, const size_t size, 
                            const ndpp_memory::DeviceType dtype)
{
    ndpp_array_base::ArrayDevice<T>::DeviceRefer(data, size, dtype,
                                                 "Array.hpp", "Array<T>::refer()");
}

template<class T>
inline void Array<T>::refer(const Array<T> &src)
{
    refer(src.data(), src.size(), src.device());
}

template<class T>
inline void Array<T>::migrateTo(T **dst, size_t &array_size, 
                                ndpp_memory::DeviceType &dtype,
                                ndpp_memory::DeviceStatus &dstatus)
{
    dst[0] = data();
    array_size = size();
    dtype = device();
    dstatus = status();

    ndpp_array_base::ArrayDevice<T>::DeviceReset();
}

template<class T>
inline void Array<T>::allocate(const size_t size, 
                               const ndpp_memory::DeviceType dtype)
{
    ndpp_array_base::ArrayDevice<T>::DeviceAlloc(size, dtype,
                                                 "Array.hpp", "Array<T>::allocate()");
}

template<class T>
inline void Array<T>::destory()
{
    string msg;
    ndpp_array_base::ArrayDevice<T>::DeviceDeAlloc("Array.hpp", "Array<T>::destory()");
}

}; // namespace ndpp