#include <include/tensor/Tensor.hpp>
#include <include/base/DataArch.hpp>
#include <include/logging/Logging.hpp>


namespace ndpp
{

Tensor Tensor::select(int64_t row_index) const
{
    if (!data())
    {
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::Error, 
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
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::Error, 
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
        ndpp_log::logger("Tensor.cpp", "Tensor::select()", ndpp_log::RuntimeType::Error, 
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

Tensor Tensor::to(const ndpp_memory::ScalarType stype, const ndpp_memory::DeviceType dtype) const
{
    Tensor _dst;
    _dst.zerosV(sizes(), strides(), stype, dtype);

    bool _is_src_buffered = false;
    void *_src_raw_ptr = nullptr;

    switch (device())
    {
        case ndpp_memory::DeviceType::Host:
    #ifdef CUDA
        case ndpp_memory::DeviceType::CudaPinned:
        case ndpp_memory::DeviceType::CudaUnified:
    #endif
            switch (dtype)
            {
                case ndpp_memory::DeviceType::Host:
            #ifdef CUDA
                case ndpp_memory::DeviceType::CudaPinned:
                case ndpp_memory::DeviceType::CudaUnified:
                case ndpp_memory::DeviceType::CudaZeroCpy:
            #endif
                    {
                        _src_raw_ptr = data();
                    }
                    break;

            #ifdef CUDA
                case ndpp_memory::DeviceType::CudaDevice:
                    {
                        _src_raw_ptr = ndpp_memory::mixScalarAlloc(allocations(), scalar(), dtype, 
                                                                   "Tensor.cpp", "Tensor::to");

                        ndpp_memory::mixScalarCopy(data(), device(),
                                                   _src_raw_ptr, dtype, scalar(), allocations(),
                                                   "Tensor.cpp", "Tensor::to");                  
                        _is_src_buffered = true;
                    }
                    break;
            #endif
            }
            break;

    #ifdef CUDA
        case ndpp_memory::DeviceType::CudaDevice:
            switch (dtype)
            {
                case ndpp_memory::DeviceType::Host:
                case ndpp_memory::DeviceType::CudaPinned:
                case ndpp_memory::DeviceType::CudaUnified:
                case ndpp_memory::DeviceType::CudaZeroCpy:
                    {
                        _src_raw_ptr = ndpp_memory::mixScalarAlloc(allocations(), scalar(), dtype, 
                                                                   "Tensor.cpp", "Tensor::to");

                        ndpp_memory::mixScalarCopy(data(), device(),
                                                   _src_raw_ptr, dtype, scalar(), allocations(),
                                                   "Tensor.cpp", "Tensor::to");                    
                        _is_src_buffered = true;                           
                    }
                    break;

                case ndpp_memory::DeviceType::CudaDevice:
                    {
                        _src_raw_ptr = data();
                    }
                    break;
            }
            break;

        case ndpp_memory::DeviceType::CudaZeroCpy:
            switch (dtype)
            {
                case ndpp_memory::DeviceType::Host:
                case ndpp_memory::DeviceType::CudaPinned:
                case ndpp_memory::DeviceType::CudaUnified:
                case ndpp_memory::DeviceType::CudaZeroCpy:
                    {
                        _src_raw_ptr = data();
                    }
                    break;

                case ndpp_memory::DeviceType::CudaDevice:
                    {
                        _src_raw_ptr = ndpp_memory::mixScalarAlloc(allocations(), scalar(), dtype, 
                                                                   "Tensor.cpp", "Tensor::to");

                        ndpp_memory::mixScalarCopy(data(), device(),
                                                   _src_raw_ptr, dtype, scalar(), allocations(),
                                                   "Tensor.cpp", "Tensor::to");   
                        _is_src_buffered = true;
                    }
                    break;
            }
            break;
    #endif
    }

    cvtScalarType(_src_raw_ptr, scalar(), 
                  _dst.data(), stype, dtype, allocations());

    if (_is_src_buffered)
    {
        ndpp_memory::mixScalarDeAlloc(_src_raw_ptr, scalar(), dtype, "Tensor.cpp", "Tensor::to");
    }

    return _dst;
}

}; // namespace ndpp