#include <include/base/DataArch.hpp>
#include <include/base/DeviceProc.hpp>
#include <include/base/ScalarType.hpp>
#include <include/tensor/TensorBase.hpp>


// Get Info's ndpp_memory::DeviceType via data's ndpp_memory::DeviceType.
static inline ndpp::ndpp_memory::DeviceType getInfoTyByDataTy(const ndpp::ndpp_memory::DeviceType data_dtype)
{
    switch (data_dtype)
    {
        case ndpp::ndpp_memory::DeviceType::Host:
            return ndpp::ndpp_memory::DeviceType::Host;   

    #ifdef CUDA
        case ndpp::ndpp_memory::DeviceType::CudaDevice:
        case ndpp::ndpp_memory::DeviceType::CudaPinned:
        case ndpp::ndpp_memory::DeviceType::CudaUnified:
        case ndpp::ndpp_memory::DeviceType::CudaZeroCpy:
            return ndpp::ndpp_memory::DeviceType::CudaZeroCpy;
    #endif
    }

    return ndpp::ndpp_memory::DeviceType::Host;
}


namespace ndpp
{

namespace ndpp_tensor_base
{

TensorDevice::TensorDevice()
{
    DeviceReset();
}

TensorDevice::TensorDevice(TensorDevice &&src)
{
    DeviceMigrate((void**)&src._data, (size_t**)&src._shape, (size_t**)&src._strides, 
                  src._dim, src._stype, src._data_dtype, src._info_dtype, src._dstatus,
                  "TensorBase.cpp", "TensorDevice::TensorDevice(TensorDevice&&");
}

TensorDevice::TensorDevice(const TensorDevice &src)
{
    if (this == &src)
    {
        return;
    }

    switch (src.DeviceSt())
    {
    case ndpp_memory::DeviceStatus::Allocation:
        DeviceCopy(src._data, src._stype, src._data_dtype, src._data_dtype, 
                   src.DeviceShape(), src.DeviceStrides(), 
                   "TensorDevice.cpp", "TensorDevice::TensorDevice(const TensorDevice&)");
        break;
    case ndpp_memory::DeviceStatus::Reference:
        DeviceRefer(src._data, src.DeviceShape(), src.DeviceStrides(), src._stype, src._data_dtype,
                    "TensorDevice.cpp", "TensorDevice::TensorDevice(const TensorDevice&)");
        break;
    }
}

TensorDevice::~TensorDevice()
{
    DeviceDeAlloc("TensorDevice.cpp", "TensorDevice<T>::~TensorDevice()");
}

void TensorDevice::DeviceAlloc(const SizeTArray &shape, const SizeTArray &strides,
                               const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype,
                               const string &file_name, const string &method_name)
{
    DeviceDeAlloc(file_name, method_name);

    // Get actual shape & strides.
    SizeTArray _act_shape, _act_strides;
    ndpp_data_arch::calcShape(shape, strides, _act_shape, ndpp_memory::DeviceType::Host,
                              file_name, method_name);
    ndpp_data_arch::calcStrides(_act_shape, stype, _act_strides, ndpp_memory::DeviceType::Host,
                                file_name, method_name);
    
    // Get allocated size without pointer.
    size_t _act_size = ndpp_data_arch::calcAllocSize(_act_shape, _act_strides, nullptr, 
                                                     file_name, method_name);
    
    // Assign '_stype'(ndpp::ndpp_memory::ScalarType), '_data_type'(ndpp::ndpp_memory::DeviceType)
    this->_stype = stype;  
    
    // Assign '_data_type'(ndpp::ndpp_memory::DeviceType).
    this->_data_dtype = dtype;
    
    // Assign '_info_dtype'(ndpp::ndpp_memory::DeviceType), and alloacate '_dim' '_shape' & '_strides'.
    if (_act_size > 0)
    {
        this->_info_dtype = getInfoTyByDataTy(this->_data_dtype);

        this->_dim = shape.size();
        this->_shape = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                        file_name, method_name);
        this->_strides = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                            file_name, method_name);

        ndpp_memory::mixMemoryCopy(shape.data(), shape.device(),
                                   this->_shape, this->_info_dtype, this->_dim,
                                   file_name, method_name);    
        ndpp_memory::mixMemoryCopy(_act_strides.data(), _act_strides.device(),
                                   this->_strides, this->_info_dtype, this->_dim,
                                   file_name, method_name);
    }
    else
    {
        _act_size = 1;
        this->_dim = 0;
        this->_shape = nullptr;
        this->_strides = nullptr;
    }

    // Alloacate '_data'.
    this->_data = ndpp_memory::mixScalarAlloc(_act_size, this->_stype, this->_data_dtype, 
                                              file_name, method_name);

    // Assign '_dstatus'(ndpp::ndpp_memory::DeviceStatus).
    this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
}

void TensorDevice::DeviceReset()
{
    this->_data = nullptr;
    this->_shape = nullptr;
    this->_strides = nullptr;
    this->_dim = 0;

    this->_stype = ndpp_memory::ScalarType::UInt8;
    this->_data_dtype = ndpp_memory::DeviceType::Host;
    this->_info_dtype = ndpp_memory::DeviceType::Host;
    this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
}

void TensorDevice::DeviceDeAlloc(const string &file_name, const string &method_name)
{
    ndpp_memory::mixMemoryDeAlloc(this->_shape, this->_info_dtype, file_name, method_name);
    ndpp_memory::mixMemoryDeAlloc(this->_strides, this->_info_dtype, file_name, method_name);

    if (this->_data && this->_dim > 0 && 
        this->_dstatus == ndpp_memory::DeviceStatus::Allocation)
    {
        ndpp_memory::mixScalarDeAlloc(this->_data, this->_stype, this->_data_dtype, file_name, method_name);
    }
    
    DeviceReset();
}

void TensorDevice::DeviceRefer(void *data, const SizeTArray &shape, const SizeTArray &strides, 
                               const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype,
                               const string &file_name, const string &method_name)
{
    if (data == this->_data)
    {
        return;
    }

    DeviceDeAlloc(file_name, method_name);

    size_t shape_dim = shape.size();
    size_t strides_dim = strides.size();

    if (shape_dim == strides_dim)
    {
        this->_dim = shape_dim;

        this->_data = data;
        this->_stype = stype; 
        this->_data_dtype = dtype;
        this->_dstatus = ndpp_memory::DeviceStatus::Reference;

        this->_info_dtype = getInfoTyByDataTy(this->_data_dtype);

        this->_shape = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                           file_name, method_name);
        this->_strides = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                            file_name, method_name);                                                    

        ndpp_memory::mixMemoryCopy(shape.data(), shape.device(),
                                   this->_shape, this->_info_dtype, this->_dim,
                                   file_name, method_name);    
        ndpp_memory::mixMemoryCopy(strides.data(), strides.device(),
                                   this->_strides, this->_info_dtype, this->_dim,
                                   file_name, method_name);
    }   
}

void TensorDevice::DeviceMigrate(void **data, size_t **shape, size_t **strides, size_t &dim,
                                 ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, ndpp_memory::DeviceType &info_dtype, 
                                 ndpp_memory::DeviceStatus &dstatus, const string &file_name, const string &method_name)
{
    if (this->_data && data[0] && this->_data == data[0])
    {
        return;
    }
    
    if (this->_shape && shape[0] && this->_shape == shape[0])
    {
        return;
    }

    if (this->_strides && strides[0] && this->_strides == strides[0])
    {
        return;
    }

    DeviceDeAlloc(file_name, method_name);

    this->_shape = shape[0];
    this->_strides = strides[0];
    this->_data = data[0];
    this->_dim = dim;
    this->_stype = stype;
    this->_data_dtype = data_dtype;
    this->_info_dtype = info_dtype;
    this->_dstatus = dstatus;

    shape[0] = nullptr;
    strides[0] = nullptr;
    data[0] = nullptr;
    dim = 0;
    stype = ndpp_memory::ScalarType::UInt8;
    data_dtype = ndpp_memory::DeviceType::Host;
    info_dtype = ndpp_memory::DeviceType::Host;
    dstatus = ndpp_memory::DeviceStatus::Allocation;
}

void TensorDevice::DeviceCopy(const void *data, const ndpp_memory::ScalarType src_stype, 
                              const ndpp_memory::DeviceType src_dtype, 
                              const ndpp_memory::DeviceType dst_dtype, 
                              const SizeTArray &shape, const SizeTArray &strides, 
                              const string &file_name, const string &method_name)
{
    DeviceDeAlloc(file_name, method_name);

    if (!data)
    {
        return;
    }

    size_t shape_dim = shape.size();
    size_t strides_dim = strides.size();

    if (shape_dim == strides_dim)
    {
        this->_dim = shape_dim;
        this->_stype = src_stype; 
        this->_data_dtype = dst_dtype;
        this->_info_dtype = getInfoTyByDataTy(this->_data_dtype);
        this->_dstatus = ndpp_memory::DeviceStatus::Allocation;
        
        // Get allocated size with 'src' ptr.
        size_t _act_size = ndpp_data_arch::calcAllocSize(shape, strides, data, 
                                                         file_name, method_name);   

        this->_data = ndpp_memory::mixScalarAlloc(_act_size, this->_stype, this->_data_dtype, 
                                                  file_name, method_name);
        this->_shape = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                           file_name, method_name);
        this->_strides = ndpp_memory::mixMemoryAlloc<size_t>(this->_dim, this->_info_dtype, 
                                                            file_name, method_name);

        ndpp_memory::mixScalarCopy(data, src_dtype, 
                                   this->_data, this->_data_dtype, this->_stype, _act_size,
                                   file_name, method_name);

        ndpp_memory::mixMemoryCopy(shape.data(), shape.device(),
                                   this->_shape, this->_info_dtype, this->_dim,
                                   file_name, method_name);    
        ndpp_memory::mixMemoryCopy(strides.data(), strides.device(),
                                   this->_strides, this->_info_dtype, this->_dim,
                                   file_name, method_name);
    }
}


}; // namespace ndpp::ndpp_tensor_base

}; // namespace ndpp