#pragma once

#include <include/base/DataProc.hpp>
#include <include/base/DataArch.hpp>
#include <include/base/DeviceProc.hpp>
#include <include/logging/Logging.hpp>
#include <include/tensor/TensorBase.hpp>
#ifdef HALF
#include <include/half/half.hpp>
#endif
#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif



namespace ndpp
{

/*
    Tensor: Representing a multi-dimensional array with a specific scalar type and device.
    Reference: https://docs.pytorch.org/cppdocs/api/aten/tensor.html
*/    
class Tensor : private ndpp_tensor_base::TensorDevice
{
public:
    Tensor() = default;
    virtual ~Tensor() = default;

    inline Tensor(const Tensor &src);

    /* 
        Used for the specified situation.
        Tensor dst;
        dst = func(...);
    */
    inline Tensor& operator=(const Tensor &src);

    // Access specified element.
    inline Tensor operator[](int64_t index) const;


    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type   src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type  src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type  src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type  src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type    src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type   src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type   src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type   src);
#ifdef HALF
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type src);
#endif
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src);
    inline Tensor& operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type    src);


    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type()   const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type()  const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type()  const; 
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type()  const; 
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type()    const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type()   const; 
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type()   const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type()   const;
#ifdef HALF
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type() const;
#endif
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type() const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type() const;
    inline operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type()    const;
    
    
    // Direct access to the underlying contiguous storage.
    inline void* data() const;

    // Get the dimension.
    inline size_t dim() const;

    // Get the element size on acutal use.
    // P.S elements() isn't necessarily same as allocations().
    inline size_t elements() const;

    // Get the actual size on memory allocation.
    // P.S allocations() isn't necessarily same as elements().
    inline size_t allocations() const;
    
    // Get the shape.
    inline SizeTArray sizes() const; 

    // Get the strides.
    inline SizeTArray strides() const; 

    // Get the ndpp_memory::ScalarType (data type).
    inline ndpp_memory::ScalarType scalar() const;

    // Get the ndpp_memory::DeviceType (memory type).
    inline ndpp_memory::DeviceType device() const;

    // Get the ndpp_memory::DeviceStatus (allocation status).
    inline ndpp_memory::DeviceStatus status() const; 

    // Copy other Tensor to itself via ndpp_memory::DeviceType.
    inline void copy(const Tensor &src, const ndpp_memory::DeviceType dtype);

    // Clone itself to new Tensor via ndpp_memory::DeviceType.
    inline Tensor clone(const ndpp_memory::DeviceType dtype) const;

    // Slicing the input tensor along the selected dimension at the given index.
    // Reference:
    //  - https://docs.pytorch.org/docs/stable/generated/torch.select.html#torch.select
    //  - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/templates/TensorBody.h#L327C12-L327C18
    Tensor select(int64_t row_index) const;

    // Reference the data from other via pointer.
    inline void refer(void *data, const SizeTArray &shape, const SizeTArray &strides, 
                      const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype);

    // Reference the data from other via ndpp::Tensor.
    inline void refer(const Tensor &src);

    // Migrate from data pointers (data, shape & strides).
    // P.S It will reset all pointers which comes from input to nullptr after pointer migration.
    inline void migrate(void **data, size_t **shape, size_t **strides, const size_t dim,
                        ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, 
                        ndpp_memory::DeviceType &info_dtype, ndpp_memory::DeviceStatus &dstatus);

    // Migrate from data pointers (data, shape & strides).
    // P.S It will reset all pointers which comes from input to nullptr after pointer migration.
    inline void migrate(void **data, SizeTArray &shape, SizeTArray &strides,
                        ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, 
                        ndpp_memory::DeviceStatus &dstatus);                    
    
    // Allocate the memory with shape, strides, different ndpp_memory::ScalarType & different ndpp_memory::DeviceType.
    inline void zerosV(const SizeTArray &shape, const SizeTArray &strides,
                       const ndpp_memory::ScalarType stype, 
                       const ndpp_memory::DeviceType dtype);
    
    // Allocate the memory with shape, different ndpp_memory::ScalarType & different ndpp_memory::DeviceType.
    // P.S The stride will be calculated via shape, and the allocation will be moved to zerosV().
    inline void zerosB(const SizeTArray &shape, const ndpp_memory::ScalarType stype, 
                       const ndpp_memory::DeviceType dtype);

    // Deallocate the memory.
    inline void destory();


private:
    // Convert single element to specific scalar type.
    // P.S It must be 0-dim and has single element inside.
    template<typename T>
    inline T toItem(const string &operator_name) const;
};

inline Tensor::Tensor(const Tensor &src)
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

inline Tensor& Tensor::operator=(const Tensor &src)
{
    copy(src, src.device());
    return *this;
}

inline Tensor Tensor::operator[](int64_t index) const
{
    return select(index);
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::UInt8, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>)");   
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::UInt16, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::UInt32, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::UInt64, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Int8, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Int16, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Int32, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Int64, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>)");
    return *this;
}

#ifdef HALF

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Float16, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type)");
    return *this;
}

#endif

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Float32, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Float64, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>)");
    return *this;
}

inline Tensor& Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type src)
{
    ndpp_memory::memset(data(), &src, scalar(), ndpp_memory::ScalarType::Bool, device(), allocations(), 
                        "Tensor.hpp", "Tensor::operator=(const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>)");
    return *this;
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>");
} 

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>");
}
    
inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>");
}

#ifdef HALF

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>");
}

#endif

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>");
}

inline Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type() const
{
    return toItem<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>("Tensor::operator ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>");
}

inline void* Tensor::data() const
{
    return DevicePtr();
}

inline size_t Tensor::dim() const
{
    return DeviceDim();
}

inline size_t Tensor::elements() const
{
    return ndpp_data_arch::calcEleSize(sizes(), data());
}

inline size_t Tensor::allocations() const
{
    return ndpp_data_arch::calcAllocSize(sizes(), strides(), data(),
                                         "Tensor.hpp", "Tensor::allocations()");
}

inline SizeTArray Tensor::sizes() const
{
    return DeviceShape();
}

inline SizeTArray Tensor::strides() const
{
    return DeviceStrides();
}

inline ndpp_memory::ScalarType Tensor::scalar() const
{
    return DeviceSca();
}

inline ndpp_memory::DeviceType Tensor::device() const
{
    return DeviceTy();
}

inline ndpp_memory::DeviceStatus Tensor::status() const
{
    return DeviceSt();
}

inline void Tensor::copy(const Tensor &src, const ndpp_memory::DeviceType dtype)
{
    DeviceCopy(src.data(), src.scalar(), src.device(), dtype, 
               src.sizes(), src.strides(), "Tensor.hpp", "Tensor::copy()");
}

inline Tensor Tensor::clone(const ndpp_memory::DeviceType dtype) const
{
    Tensor _dst;
    _dst.copy(*this, dtype);
    return _dst;
}

inline void Tensor::refer(void *data, const SizeTArray &shape, const SizeTArray &strides, 
                          const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype)
{
    DeviceRefer(data, shape, strides, stype, dtype,
                "Tensor.hpp", "Tensor::zerosV()");
}                          

inline void Tensor::refer(const Tensor &src)
{
    refer(src.data(), src.sizes(), src.strides(), 
          src.scalar(), src.device());
}

inline void Tensor::migrate(void **data, size_t **shape, size_t **strides, const size_t dim,
                            ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, 
                            ndpp_memory::DeviceType &info_dtype, ndpp_memory::DeviceStatus &dstatus)
{
    DeviceMigrate(data, shape, strides, dim,
                  stype, data_dtype, info_dtype, dstatus, 
                  "Tensor.hpp", "Tensor::migrate(size_t**)");
}

inline void Tensor::migrate(void **data, SizeTArray &shape, SizeTArray &strides,
                            ndpp_memory::ScalarType &stype, ndpp_memory::DeviceType &data_dtype, 
                            ndpp_memory::DeviceStatus &dstatus)
{
    if (shape.size() != strides.size())
    {
        ndpp_log::logger("Tensor.hpp", "Tensor::migrate(SizeTArray&)", ndpp_log::RuntimeType::WARN, 
                         "shape.size() differs from strides.size().", true);
        return;
    }
    if (shape.device() != strides.device())
    {
        ndpp_log::logger("Tensor.hpp", "Tensor::migrate(SizeTArray&)", ndpp_log::RuntimeType::WARN, 
                         "shape.device() differs from strides.device().", true);
        return;
    }
    if (shape.status() != strides.status())
    {
        ndpp_log::logger("Tensor.hpp", "Tensor::migrate(SizeTArray&)", ndpp_log::RuntimeType::WARN, 
                         "shape.status() differs from strides.status().", true);
        return;
    }

    size_t dim = 0;
    size_t *_shape = nullptr, *_strides = nullptr;
    ndpp_memory::DeviceType info_dtype = ndpp_memory::DeviceType::Host;
    ndpp::ndpp_memory::DeviceStatus info_dstatus = ndpp::ndpp_memory::DeviceStatus::Allocation;
    shape.migrateTo((size_t**)&_shape, dim, info_dtype, info_dstatus);
    strides.migrateTo((size_t**)&_strides, dim, info_dtype, info_dstatus);

    DeviceMigrate(data, (size_t**)&_shape, (size_t**)&_strides, dim,
                  stype, data_dtype, info_dtype, dstatus, 
                  "Tensor.hpp", "Tensor::migrate(SizeTArray&)");
}

inline void Tensor::zerosV(const SizeTArray &shape, const SizeTArray &strides,
                           const ndpp_memory::ScalarType stype, 
                           const ndpp_memory::DeviceType dtype)
{
    DeviceAlloc(shape, strides, stype, dtype, 
                "Tensor.hpp", "Tensor::zerosV()");
}

void Tensor::zerosB(const SizeTArray &shape, const ndpp_memory::ScalarType stype, 
                    const ndpp_memory::DeviceType dtype)
{
    SizeTArray strides;
    ndpp_data_arch::calcStrides(shape, stype, strides, ndpp_memory::DeviceType::Host,
                                "Tensor.hpp", "Tensor::zerosB()");

    zerosV(shape, strides, stype, dtype);
}

void Tensor::destory()
{
    DeviceDeAlloc("Tensor.hpp", "Tensor::destory()");
}

template<typename T>
inline T Tensor::toItem(const string &operator_name) const
{
    if (dim() > 0)
    {
        ndpp_log::logger("Tensor.hpp", "Tensor::toItem()", ndpp_log::RuntimeType::ERROR, 
                         "Only one element tensors can be converted to Python scalars or C++ arithmetic types.", true);
        exit(EXIT_FAILURE);
    }

    return ndpp_memory::scalarPtrAccess<T>(data(), scalar(), device(), 0, "Tensor.hpp", "Tensor::toItem()");
}


}; // namespace ndpp