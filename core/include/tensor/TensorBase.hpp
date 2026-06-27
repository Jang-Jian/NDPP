#ifndef __TensorBase_hpp__
#define __TensorBase_hpp__

#include <iostream>

#include <include/base/DeviceProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/array/ArrayStype.hpp>

using namespace std;


namespace ndpp
{

namespace ndpp_tensor_base
{


/*
    TensorDevice: Used for ndpp::Tensor, which processes memory processing.
    
    P.S The default allocation TensorDevice() has these the settings in below.
    _data = nullptr.
    _shape = nullptr.
    _strides = nullptr.
    _dim = 0
    _stype = ndpp_memory::ScalarType::UInt8.
    _data_dtype = ndpp_memory::DeviceType::Host.
    _info_dtype = ndpp_memory::DeviceType::Host.
    _dstatus = ndpp_memory::DeviceStatus::Allocation.
*/
class TensorDevice
{
public:
    TensorDevice();
    TensorDevice(TensorDevice &&src);
    TensorDevice(const TensorDevice &src);
    
    virtual ~TensorDevice();

protected:
    void DeviceAlloc(const SizeTArray &shape, const SizeTArray &strides,
                     const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype,
                     const string &file_name, const string &method_name);

    void DeviceReset();

    void DeviceDeAlloc(const string &file_name, const string &method_name);

    void DeviceRefer(void *data, const SizeTArray &shape, const SizeTArray &strides, 
                     const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype,
                     const string &file_name, const string &method_name);

    void DeviceMigrate(void **data, size_t **shape, size_t **strides, size_t &dim,
                       ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, ndpp_memory::DeviceType &info_dtype, 
                       ndpp_memory::DeviceStatus &dstatus, const string &file_name, const string &method_name);

    void DeviceCopy(const void *data, const ndpp_memory::ScalarType src_stype, 
                    const ndpp_memory::DeviceType src_dtype, 
                    const ndpp_memory::DeviceType dst_dtype, 
                    const SizeTArray &shape, const SizeTArray &strides, 
                    const string &file_name, const string &method_name);

    inline void* DevicePtr() const;
    inline size_t DeviceDim() const;
    inline SizeTArray DeviceShape() const;
    inline SizeTArray DeviceStrides() const;
    inline ndpp_memory::ScalarType DeviceSca() const;
    inline ndpp_memory::DeviceType DeviceTy() const;
    inline ndpp_memory::DeviceStatus DeviceSt() const;

private:
    void *_data = nullptr;
    size_t *_shape = nullptr;
    size_t *_strides = nullptr;
    size_t _dim = 0;

    ndpp_memory::ScalarType _stype = ndpp_memory::ScalarType::UInt8;
    ndpp_memory::DeviceType _data_dtype = ndpp_memory::DeviceType::Host;
    ndpp_memory::DeviceType _info_dtype = ndpp_memory::DeviceType::Host;
    ndpp_memory::DeviceStatus _dstatus = ndpp_memory::DeviceStatus::Allocation;
};

inline void* TensorDevice::DevicePtr() const
{
    return this->_data;
}

inline size_t TensorDevice::DeviceDim() const
{ 
    return this->_dim;
}

inline SizeTArray TensorDevice::DeviceShape() const
{
    SizeTArray dst_shape;
    dst_shape.refer(this->_shape, this->_dim, this->_info_dtype);
    return dst_shape;
}

inline SizeTArray TensorDevice::DeviceStrides() const
{
    SizeTArray dst_strides;
    dst_strides.refer(this->_strides, this->_dim, this->_info_dtype);
    return dst_strides;
}

inline ndpp_memory::ScalarType TensorDevice::DeviceSca() const
{
    return this->_stype;
}

inline ndpp_memory::DeviceType TensorDevice::DeviceTy() const
{
    return this->_data_dtype;
}

inline ndpp_memory::DeviceStatus TensorDevice::DeviceSt() const
{
    return this->_dstatus;
}


}; // namespace ndpp::ndpp_tensor_base

}; // namespace ndpp

#endif