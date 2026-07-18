#include <include/extension/CudaProc.hpp>
#include <include/tensor/TensorArithmeticProc.hpp>
#include <include/tensor/GpuTensorArithmetic.cuh>


namespace ndpp
{

namespace ndpp_arithmetic
{

// gpuArithForwardKernel: Arithmetic forward.
// P.S The arithmetic supports these operation:
// - Calculating Tensor(a) {arithmetic} Tensor(b).
// - Calculating Tensor(a) {arithmetic} Scalar(b).
// - Calculating Scalar(a) {arithmetic} Tensor(b).
template<typename T1, typename T2>
__global__ void gpuArithForwardKernel(const T1 *a, const T2 *b, T1 *c,
                                      const size_t *a_allocated_shape, const size_t *a_acutal_shape, const int64_t a_shape_dim, const int64_t a_total,
                                      const size_t *b_allocated_shape, const size_t *b_acutal_shape, const int64_t b_shape_dim, const int64_t b_total,
                                      const int64_t loop_total, const Arithmetic arith_type)
{
    const int64_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if (index < loop_total)
    {
        T1 a_val = static_cast<T1>(0);
        T2 b_val = static_cast<T2>(0);  

        const bool available_a = arithmeticIndexing(a, index, a_total, a_shape_dim, 
                                                    a_acutal_shape, a_allocated_shape, a_val);
        const bool available_b = arithmeticIndexing(b, index, b_total, b_shape_dim, 
                                                    b_acutal_shape, b_allocated_shape, b_val);
        if (available_a && available_b)
        {
            c[index] = arithmeticOp(a_val, b_val, arith_type);
        }
    }
}


// gpuArithForwardBTypeImpleKernel: Deciding the data type for b.
// P.S Wrapping for gpuArithForwardKernel() with different data type b.
template<typename T>
static inline void gpuArithForwardBTypeImpleKernel(const T *a, const void *b, T *c, const ndpp_memory::ScalarType b_stype,
                                                   const size_t *a_allocated_shape, const size_t *a_acutal_shape, const int64_t a_shape_dim, const int64_t a_total,
                                                   const size_t *b_allocated_shape, const size_t *b_acutal_shape, const int64_t b_shape_dim, const int64_t b_total,
                                                   const int64_t loop_total, const Arithmetic arith_type)
{
    constexpr size_t arith_block_thread_t = 512;
    dim3 grid_set = ndpp_memory::ndpp_cuda::cudaExecGrid(loop_total, arith_block_thread_t);

    switch (b_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt64:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int8:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int16:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int32:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int64:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Float64:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Bool:
            gpuArithForwardKernel<<<grid_set, arith_block_thread_t>>>(a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(b), c,
                                                                      a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                                                      b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                                                      loop_total, arith_type);
            break;
    }

    ndpp_memory::ndpp_cuda::cudaErrorChecker(cudaPeekAtLastError(), "GpuTensorArithmeticFwd.cu", "ndpp::ndpp_arithmetic::gpuArithForwardBTypeImpleKernel()");
}


// cpuArithForwardACTypeImpleKernel: Deciding the data type for a & c.
// P.S Wrapping for gpuArithForwardBTypeImpleKernel() with different data type a & c.
void gpuArithForwardACTypeImpleKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                                      const size_t *a_allocated_shape, const size_t *a_acutal_shape, const int64_t a_shape_dim, const int64_t a_total,
                                      const size_t *b_allocated_shape, const size_t *b_acutal_shape, const int64_t b_shape_dim, const int64_t b_total,
                                      const int64_t loop_total, const Arithmetic arith_type)
{
    switch (a_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::UInt64:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int8:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int16:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int32:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Int64:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Float64:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
        case ndpp_memory::ScalarType::Bool:
            gpuArithForwardBTypeImpleKernel(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_shape_dim, b_total, 
                                            loop_total, arith_type);
            break;
    }
}


}; // namespace ndpp_arithmetic

}; // namespace ndpp