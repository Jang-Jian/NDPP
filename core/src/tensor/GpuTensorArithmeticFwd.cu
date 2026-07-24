#include <include/extension/CudaProc.hpp>
#include <include/tensor/TensorArithmeticProc.hpp>
#include <include/tensor/GpuTensorArithmetic.cuh>


namespace ndpp
{

namespace ndpp_arithmetic
{

// gpuArithForwardImpleKernel(CUDA): Arithmetic forward.
// P.S The arithmetic supports these operation:
// - Calculating Tensor(a) {arithmetic} Tensor(b).
// - Calculating Tensor(a) {arithmetic} Scalar(b).
// - Calculating Scalar(a) {arithmetic} Tensor(b).
template<typename Operator, typename T1, typename T2>
__global__ void gpuArithForwardImpleKernel(const T1 *a, const T2 *b, T1 *c,
                                           const size_t *a_allocated_shape, const size_t *a_acutal_shape, const size_t *a_strides, 
                                           const int64_t a_shape_dim, const int64_t a_total,
                                           const size_t *b_allocated_shape, const size_t *b_acutal_shape, const size_t *b_strides, 
                                           const int64_t b_shape_dim, const int64_t b_total, const int64_t loop_total)
{
    const int64_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= loop_total)
        return;
    
    T1 a_val;  
    if (!arithmeticIndexingV2(a, index, a_total, a_shape_dim, 
                              a_acutal_shape, a_allocated_shape, a_strides, a_val))
        return;
    
    T2 b_val;
    if (!arithmeticIndexingV2(b, index, b_total, b_shape_dim, 
                              b_acutal_shape, b_allocated_shape, b_strides, b_val))
        return;

    c[index] = Operator()(a_val, b_val);
}


// gpuArithForwardBTypeImpleKernel: Deciding the data type for b.
// P.S Wrapping for gpuArithForwardImpleKernel() with different data type b.
template<typename Operator, typename T>
static inline void gpuArithForwardBTypeImpleKernel(const T *a, const void *b, T *c, const ndpp_memory::ScalarType b_stype,
                                                   const size_t *a_allocated_shape, const size_t *a_acutal_shape, const size_t *a_strides, 
                                                   const int64_t a_shape_dim, const int64_t a_total,
                                                   const size_t *b_allocated_shape, const size_t *b_acutal_shape, const size_t *b_strides,
                                                   const int64_t b_shape_dim, const int64_t b_total, const int64_t loop_total)
{
    constexpr size_t arith_block_thread_t = 512;
    dim3 grid_set = ndpp_memory::ndpp_cuda::cudaExecGrid(loop_total, arith_block_thread_t);

    switch (b_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::UInt64:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Int8:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Int16:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Int32:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Int64:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
    #endif
        case ndpp_memory::ScalarType::Float32:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Float64:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
        case ndpp_memory::ScalarType::Bool:
            gpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type><<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                                                      loop_total);
            break;
    }

    ndpp_memory::ndpp_cuda::cudaErrorChecker(cudaPeekAtLastError(), "GpuTensorArithmeticFwd.cu", "ndpp::ndpp_arithmetic::gpuArithForwardBTypeImpleKernel()");
}


// gpuArithForwardACTypeImpleKernel: Deciding the data type for a & c.
// P.S Wrapping for gpuArithForwardBTypeImpleKernel() with different data type a & c.
template<typename Operator>
void gpuArithForwardACTypeImpleKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                                      const size_t *a_allocated_shape, const size_t *a_acutal_shape, const size_t *a_strides, 
                                      const int64_t a_shape_dim, const int64_t a_total,
                                      const size_t *b_allocated_shape, const size_t *b_acutal_shape, const size_t *b_strides, 
                                      const int64_t b_shape_dim, const int64_t b_total, const int64_t loop_total)
{
    switch (a_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt64:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int8:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int16:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int32:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int64:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    #endif           
        case ndpp_memory::ScalarType::Float32:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Float64:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Bool:
            gpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    }
}


void gpuArithForwardKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                           const size_t *a_allocated_shape, const size_t *a_acutal_shape, const size_t *a_strides, int64_t a_shape_dim, int64_t a_total,
                           const size_t *b_allocated_shape, const size_t *b_acutal_shape, const size_t *b_strides, int64_t b_shape_dim, int64_t b_total,
                           int64_t loop_total, Arithmetic arith_type)
{
    switch (arith_type)
    {
    case Arithmetic::Add:
        gpuArithForwardACTypeImpleKernel<AddOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Subtract:
        gpuArithForwardACTypeImpleKernel<SubOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Multiply:
        gpuArithForwardACTypeImpleKernel<MulOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Division:
        gpuArithForwardACTypeImpleKernel<DivOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    }
}


}; // namespace ndpp_arithmetic

}; // namespace ndpp