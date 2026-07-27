#include <include/tensor/Tensor.hpp>
#include <include/base/DataArch.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

Tensor Tensor::select(int64_t row_index) const
{
    if (!data())
    {
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::ERROR, 
                         "The tensor is empty, please allocate it first.", true);
        exit(EXIT_FAILURE);
    }

    
    IntArray _shape = sizes();
    IntArray _strides = strides();

    //IntArray _act_shape;
    //ndpp_data_arch::calcShape(_shape, _strides, _act_shape, ndpp_memory::DeviceType::Host, 
        //                      "Tensor.cpp", "Tensor::select()");    
    
    size_t _dim = dim();
    if (_dim == 0)
    {
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::ERROR, 
                         "It can't be applied to a 0-dim tensor.", true);
        exit(EXIT_FAILURE);
    }

    IntArray _return_strides, _return_shape;
    //IntArray _return_act_shape, _return_shape;
    if (_dim - 1 > 0)
    {
        _return_shape.refer(_shape.data() + 1, _dim - 1, _shape.device());
        _return_strides.refer(_strides.data() + 1, _dim - 1, _strides.device());
        //_return_shape.allocate(_dim - 1, _shape.device()); 
        //_return_act_shape.allocate(_dim - 1, ndpp_memory::DeviceType::Host); 
    }

    Integer actual_col_size = 1;
    for (size_t _idx = 1; _idx < _dim; ++_idx)
    {
        //actual_col_size *= _act_shape[_idx];
        actual_col_size *= _strides[_idx - 1] / _strides[_idx];
        //_return_act_shape[_idx - 1] = _act_shape[_idx];
        //_return_shape[_idx - 1] = _shape[_idx];
    }

    // Adjust the negative index, when _dim > 0.
    Integer shape_1st = _shape[0];
    if (row_index < 0)
    {
        row_index += shape_1st;
        //row_index += static_cast<int64_t>(_act_shape[0]);
    }


    // judge the row_index whether to be negative or over size.
    if (row_index < 0 || row_index >= shape_1st)
    {
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::ERROR, 
                         "Index " + to_string(static_cast<long long>(row_index)) + " is out of bounds with size " + 
                         to_string(static_cast<long long>(shape_1st)) + ".", true);
        exit(EXIT_FAILURE);
    }


    ndpp::ndpp_memory::ScalarType data_stype = scalar();
    ndpp::ndpp_memory::DeviceType data_dtype = device();


    
    /*ndpp_data_arch::calcStrides(_return_act_shape, data_stype, 
                                _return_strides, _strides.device(),
                                "Tensor.cpp", "Tensor::select()");*/


    // get the start address of pointer with row_index, using 2-dimension access.
    void *_return_data = ndpp_memory::scalarPtrShift(data(), data_stype, (row_index * actual_col_size));
    //ndpp::ndpp_memory::DeviceStatus dstatus = ndpp::ndpp_memory::DeviceStatus::Reference;

    Tensor dst;
    dst.refer(_return_data, _return_shape, _return_strides, data_stype, data_dtype);
    //dst.migrate((void**)&_return_data, _return_shape, _return_strides,
    //            data_stype, data_dtype, dstatus);

    return dst;
}

}; // namespace ndpp