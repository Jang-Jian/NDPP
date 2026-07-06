#include <include/base/DataArch.hpp>
#include <include/base/ScalarType.hpp>
#include <include/logging/Logging.hpp>


// Detect wether has same dimension or 0-dimension with 'shape' & 'strides'.
static inline bool isDimError(const ndpp::SizeTArray &shape, const ndpp::SizeTArray &strides, 
                              const string &who_call, const string &file_name, const string &method_name)
{
    const size_t shape_dim = shape.size();
    const size_t strides_dim = strides.size();

    if (shape_dim != strides_dim)
    {
        ndpp::ndpp_log::logger(file_name, method_name, ndpp::ndpp_log::RuntimeType::WARN, 
                               "In " + who_call + ", the shape's dim differs from strides's dim.", true);
        return false;
    }

    if (shape_dim == 0)
    {
        return false;
    }

    return true;
}


// Used for 0-dim detection.
static inline size_t isZeroDim(const void *data_ptr)
{
    // Single data & 0-dim.
    if (data_ptr)
    {
        return 1;
    }

    return 0;
}


namespace ndpp
{

namespace ndpp_data_arch
{


void calcShape(const SizeTArray &src_shape, const SizeTArray &src_strides, 
               SizeTArray &dst_shape, const ndpp_memory::DeviceType dst_shape_dtype,
               const string &file_name, const string &method_name)
{
    if (!isDimError(src_shape, src_strides, "ndpp_data_arch::calcShape()", 
                    file_name, method_name))
    {
        dst_shape.destory();
        return;
    }

    dst_shape.allocate(src_shape.size(), dst_shape_dtype);

    dst_shape[0] = src_shape[0];
    for (int index = 1; index < static_cast<int>(dst_shape.size()); ++index)
    {
        if (src_strides[index - 1] > 0 && src_strides[index] > 0)
        {
            dst_shape[index] = src_strides[index - 1] / src_strides[index];
        }
        else
        {
            dst_shape[index] = 0;
        }
    }
}


void calcStrides(const SizeTArray &src_shape, const ndpp_memory::ScalarType stype,
                 SizeTArray &dst_strides, const ndpp_memory::DeviceType dst_strides_dtype,
                 const string &file_name, const string &method_name)
{
    
    size_t shape_dim = src_shape.size();

    if (src_shape.size() == 0)
    {
        dst_strides.destory();
        return;
    }

    dst_strides.allocate(src_shape.size(), dst_strides_dtype);
    size_t stype_size = ndpp_memory::sizeOfScalar(stype, file_name, method_name);

    size_t mul_val = src_shape[0];

    for (int idx = 1; idx < (int)shape_dim; ++idx)
    {
        mul_val *= src_shape[idx];
    }
    
    //cout << "mul_val: " << mul_val << endl;

    /*for (int cur_idx = 0; cur_idx < (int)shape_dim - 1; ++cur_idx)
    {
        unsigned int stride = stype_size;

        for (int nex_idx = cur_idx + 1; nex_idx < (int)shape_dim; ++nex_idx)
        {
            stride *= src_shape[nex_idx];
        }

        dst_strides[cur_idx] = stride;
    }*/


    unsigned int stride = stype_size;

    for (int i = shape_dim - 2; i >= 0; --i)
    {
        stride *= src_shape[i + 1];
        dst_strides[i] = stride;
    }


    if (mul_val > 0)
    {
        dst_strides[shape_dim - 1] = stype_size;
    }
    else
    {
        dst_strides[shape_dim - 1] = 0;
    }
}


size_t calcAllocSize(const SizeTArray &src_shape, const SizeTArray &src_strides, const void *data_ptr,
                     const string &file_name, const string &method_name)
{
    if (!isDimError(src_shape, src_strides, "ndpp_data_arch::calcAllocSize()", 
                    file_name, method_name))
    {
        return isZeroDim(data_ptr);
    }

    SizeTArray act_shape;
    calcShape(src_shape, src_strides, act_shape, ndpp_memory::DeviceType::Host,
              file_name, method_name);

    size_t total = act_shape[0];
    for (int index = 1; index < (int)src_shape.size(); ++index)
    {
        total *= act_shape[index];
    }

    return total;
}


size_t calcEleSize(const SizeTArray &src_shape, const void *data_ptr)
{
    if (src_shape.size() == 0)
    {
        return isZeroDim(data_ptr);
    }

    unsigned int total = 1;
    for (int index = 0; index < (int)src_shape.size(); ++index)
    {
        total *= src_shape[index];
    }

    return total;
}


}; // namespace ndpp::ndpp_data_arch

}; // namespace ndpp