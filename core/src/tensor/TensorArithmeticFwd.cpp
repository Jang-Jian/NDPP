#include <include/logging/Logging.hpp>
#include <include/base/Parallelism.hpp>
#include <include/base/DataArch.hpp>
#include <include/array/ArrayStype.hpp>
#include <include/tensor/Tensor.hpp>
#include <include/tensor/TensorArithmetic.hpp>
#include <include/tensor/TensorArithmeticProc.hpp>
#ifdef CUDA
#include <include/tensor/GpuTensorArithmetic.cuh>
#endif


namespace ndpp
{

namespace ndpp_arithmetic
{

static inline std::string shapeStr(const IntArray &shape)
{
    std::string shape_str = "[";

    for (int index = 0; index < (int)shape.size(); ++index)
    {
        if (index < (int)shape.size() - 1)
        {
            shape_str += to_string(shape[index]) + ", "; 
        }
        else
        {
            shape_str += to_string(shape[index]) + "]"; 
        }
    }

    return shape_str;
}


static inline void compareDim(const Tensor &a, const Tensor &b, 
                              const int min_dim, const string &operator_name)
{
    int issue_dim = -1;
    int issue_a_size = -1, issue_b_size = -1;

    const IntArray a_shape = a.sizes();
    const IntArray b_shape = b.sizes();
    const size_t a_dim = a_shape.size();
    const size_t b_dim = b_shape.size();


    for (int index = min_dim - 1; index >= 0; --index)
    {
        //cout << "min_dim: " << min_dim << ", " << a_shape_ptr[a_dim - 1 -index] << ", " << b_shape_ptr[b_dim - 1 -index] << endl;

        if (a_shape[a_dim - 1 - index] != b_shape[b_dim - 1 - index])
        {
            issue_dim = index;
            issue_a_size = a_shape[index];
            issue_b_size = b_shape[index];
            break;
        }
    }


    if (issue_dim >= 0)
    {
        if (a.allocations() != 1 || b.allocations() != 1)
        {
            ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                             "The size of tensor a (" + to_string(issue_a_size) + ") must match the size of tensor b (" + 
                             to_string(issue_b_size) + ") at non-singleton dimension " + to_string(issue_dim), true);
            exit(EXIT_FAILURE);
        }
    }
}


// loopTotal: parse the maximum size from tensor a (a_total) & tensor b (b_total) for looping.
static inline int64_t loopTotal(const int64_t a_total, const int64_t b_total)
{
    if (a_total == 1 || b_total == 1)
    {
        return (a_total >= b_total) ? a_total : b_total;
    }
    else if (a_total == 1 && b_total == 1)
    {
        return 1;
    }
    
    return (a_total >= b_total) ? a_total : b_total;
}


// cpuArithForwardKernel(OpenMP): Arithmetic forward.
// P.S The arithmetic supports these operation:
// - Calculating Tensor(a) {arithmetic} Tensor(b).
// - Calculating Tensor(a) {arithmetic} Scalar(b).
// - Calculating Scalar(a) {arithmetic} Tensor(b).
template<typename Operator, typename T1, typename T2>
static inline void cpuArithForwardImpleKernel(const T1* __restrict__ a, 
                                              const T2* __restrict__ b, 
                                              T1* __restrict__ c,
                                              const Integer *a_allocated_shape, const Integer *a_acutal_shape, const Integer *a_strides, 
                                              int64_t a_shape_dim, int64_t a_total,
                                              const Integer *b_allocated_shape, const Integer *b_acutal_shape, const Integer *b_strides, 
                                              int64_t b_shape_dim, int64_t b_total,
                                              int64_t loop_total)
{
    #pragma omp parallel for simd schedule(guided) num_threads(threadsReq(loop_total))
    for (int64_t index = 0; index < loop_total; ++index)
    {
        T1 a_val;
        if (!arithmeticIndexingV2(a, index, a_total, a_shape_dim, 
                                  a_acutal_shape, a_allocated_shape, a_strides, a_val))
            continue;

        T2 b_val;
        if (!arithmeticIndexingV2(b, index, b_total, b_shape_dim, 
                                  b_acutal_shape, b_allocated_shape, b_strides, b_val))
            continue;

        c[index] = Operator()(a_val, b_val);
    }
}


// cpuArithForwardBTypeImpleKernel: Deciding the data type for b.
// P.S Wrapping for cpuArithForwardImpleKernel() with different data type b.
template<typename Operator, typename T>
static inline void cpuArithForwardBTypeImpleKernel(const T *a, const void *b, T *c, const ndpp_memory::ScalarType b_stype,
                                                   const Integer *a_allocated_shape, const Integer *a_acutal_shape, const Integer *a_strides, 
                                                   int64_t a_shape_dim, int64_t a_total,
                                                   const Integer *b_allocated_shape, const Integer *b_acutal_shape, const Integer *b_strides, 
                                                   int64_t b_shape_dim, int64_t b_total, int64_t loop_total)
{
    switch (b_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            cpuArithForwardImpleKernel<Operator,T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt64:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int8:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int16:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                  loop_total);
            break;
        case ndpp_memory::ScalarType::Int32:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int64:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Float64:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Bool:
            cpuArithForwardImpleKernel<Operator, T, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(
                                  a, static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(b), c,
                                  a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                  b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    }
}


// cpuArithForwardACTypeImpleKernel: Deciding the data type for a & c.
// P.S Wrapping for cpuArithForwardBTypeImpleKernel() with different data type a & c.
template<typename Operator>
static inline void cpuArithForwardACTypeImpleKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                                                    const Integer *a_allocated_shape, const Integer *a_acutal_shape, const Integer *a_strides, 
                                                    int64_t a_shape_dim, int64_t a_total,
                                                    const Integer *b_allocated_shape, const Integer *b_acutal_shape, const Integer *b_strides, 
                                                    int64_t b_shape_dim, int64_t b_total, int64_t loop_total)
{
    switch (a_stype)
    {
        case ndpp_memory::ScalarType::UInt8:  
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt16: 
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type>(
                static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt32: 
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::UInt64:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::UInt64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int8:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int8>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int16:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type>(static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int32:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Int64:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Int64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;

    #ifdef HALF
        case ndpp_memory::ScalarType::Float16:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float16>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
    #endif
            
        case ndpp_memory::ScalarType::Float32:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float32>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, loop_total);
            break;
        case ndpp_memory::ScalarType::Float64:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Float64>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                            loop_total);
            break;
        case ndpp_memory::ScalarType::Bool:
            cpuArithForwardBTypeImpleKernel<Operator, ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type>(
                                            static_cast<const ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(a), 
                                            b, static_cast<ndpp_memory::ScalarTypeToCppType<ndpp_memory::ScalarType::Bool>::type*>(c), b_stype,
                                            a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                            b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total, 
                                            loop_total);
            break;
    }
}


static inline void cpuArithForwardKernel(const void *a, const void *b, void *c, const ndpp_memory::ScalarType a_stype, const ndpp_memory::ScalarType b_stype,
                                         const Integer *a_allocated_shape, const Integer *a_acutal_shape, const Integer *a_strides, 
                                         int64_t a_shape_dim, int64_t a_total,
                                         const Integer *b_allocated_shape, const Integer *b_acutal_shape, const Integer *b_strides, 
                                         int64_t b_shape_dim, int64_t b_total,
                                         int64_t loop_total, Arithmetic arith_type)
{
    switch (arith_type)
    {
    case Arithmetic::Add:
        cpuArithForwardACTypeImpleKernel<AddOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Subtract:
        cpuArithForwardACTypeImpleKernel<SubOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Multiply:
        cpuArithForwardACTypeImpleKernel<MulOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    case Arithmetic::Division:
        cpuArithForwardACTypeImpleKernel<DivOp>(a, b, c, a_stype,b_stype,
                                                a_allocated_shape, a_acutal_shape, a_strides, a_shape_dim, a_total,
                                                b_allocated_shape, b_acutal_shape, b_strides, b_shape_dim, b_total,
                                                loop_total);
        break;
    }
}

void arithmeticForward(const Tensor &a, const Tensor &b, Tensor &c,
                       const Arithmetic arith_type, const string &operator_name)
{
    if (!a.data())
    {
        ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                         "The tensor a is empty, and please allocate tensor a first.", true);
        exit(EXIT_FAILURE);
    }

    if (!b.data())
    {
        ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                         "The tensor b is empty, and please allocate tensor b first.", true);
        exit(EXIT_FAILURE);
    }

    const IntArray a_acutal_shape = a.sizes();
    const IntArray b_acutal_shape = b.sizes();
    const IntArray a_strides = a.strides();
    const IntArray b_strides = b.strides();

    const int64_t a_dim = static_cast<int64_t>(a_acutal_shape.size());
    const int64_t b_dim = static_cast<int64_t>(b_acutal_shape.size());
    const int64_t a_total = static_cast<int64_t>(a.allocations());
    const int64_t b_total = static_cast<int64_t>(b.allocations());
    const int64_t loop_total = loopTotal(a_total, b_total);

    const ndpp_memory::DeviceType a_dtype = a.device();
    const ndpp_memory::ScalarType a_stype = a.scalar();
    const ndpp_memory::ScalarType b_stype = b.scalar();

    if (!same(a.sizes(), c.sizes()) || a_dtype != c.device() || a_stype != c.scalar())
    {
        if (a_total >= b_total)
        {
            c.zerosV(a_acutal_shape, a_strides, a_stype, a_dtype);
        }
        else
        {
            c.zerosV(b_acutal_shape, b_strides, a_stype, a_dtype);
        }
    }


    if (a.data() != c.data())
    {
        // Used for c = a {arithmetic} b.
        const int64_t min_dim = (a_dim <= b_dim) ? static_cast<int64_t>(a_dim) : static_cast<int64_t>(b_dim);
        compareDim(a, b, min_dim, operator_name);
    }
    else
    {
        // Used for a += b, a -= b, a *= b & a /= b, and a == c.
        // So, a's dim must be bigger than b's dim.
        if (a_dim < b_dim)
        {
            std::string a_shape_str = shapeStr(a_acutal_shape);
            std::string b_shape_str = shapeStr(b_acutal_shape);
            
            ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                                "Output with shape " + a_shape_str + " doesn't match the broadcast shape " + b_shape_str + ".", true);
            exit(EXIT_FAILURE);
        }

        compareDim(a, b, b_dim, operator_name);
    }

    
    IntArray a_allocated_shape, b_allocated_shape;
    ndpp_data_arch::calcShape(a_acutal_shape, a_strides, a_allocated_shape, a_acutal_shape.device(), 
                              "TensorOperator.cpp", operator_name);
    ndpp_data_arch::calcShape(b_acutal_shape, b_strides, b_allocated_shape, a_acutal_shape.device(), 
                              "TensorOperator.cpp", operator_name);
                              
#ifdef CUDA
    const ndpp_memory::DeviceType b_dtype = b.device();

    switch (a_dtype)
    {
        case ndpp_memory::DeviceType::Host:  
        case ndpp_memory::DeviceType::CudaPinned:
        case ndpp_memory::DeviceType::CudaUnified:
        case ndpp_memory::DeviceType::CudaZeroCpy: 
            {
                switch (b_dtype)
                {
                    case ndpp_memory::DeviceType::Host:  
                    case ndpp_memory::DeviceType::CudaPinned:
                    case ndpp_memory::DeviceType::CudaUnified:
                    case ndpp_memory::DeviceType::CudaZeroCpy: 
                        {
                            cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                                  a_allocated_shape.data(), a_acutal_shape.data(), a_strides.data(), a_dim, a_total,
                                                  b_allocated_shape.data(), b_acutal_shape.data(), b_strides.data(), b_dim, b_total,
                                                  loop_total, arith_type);
                        }
                        break;
                    
                    case ndpp_memory::DeviceType::CudaDevice:
                        {
                            Tensor _b;
                            _b.copy(b, ndpp_memory::DeviceType::Host);

                            cpuArithForwardKernel(a.data(), _b.data(), c.data(), a_stype, b_stype,
                                                  a_allocated_shape.data(), a_acutal_shape.data(), a_strides.data(), a_dim, a_total,
                                                  b_allocated_shape.data(), b_acutal_shape.data(), b_strides.data(), b_dim, b_total,
                                                  loop_total, arith_type);
                        }
                        break;
                }
            } 
            break;
        
        case ndpp_memory::DeviceType::CudaDevice:  
            {
                switch (b_dtype)
                {
                    case ndpp_memory::DeviceType::Host:
                    case ndpp_memory::DeviceType::CudaUnified:
                    case ndpp_memory::DeviceType::CudaPinned:                  
                        {
                            Tensor _b;
                            _b.copy(b, ndpp_memory::DeviceType::CudaDevice);
                            
                            const Integer *a_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_acutal_shape.data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(_b.sizes().data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_strides.data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(_b.strides().data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));

                            gpuArithForwardKernel(a.data(), _b.data(), c.data(), a_stype, b_stype,
                                                  a_allocated_shape_dev, a_acutal_shape_dev, a_strides_dev, a_dim, a_total,
                                                  b_allocated_shape_dev, b_acutal_shape_dev, b_strides_dev, b_dim, b_total,
                                                  loop_total, arith_type);
                        }
                        break;

                    case ndpp_memory::DeviceType::CudaZeroCpy: 
                        {
                            const void *b_dev = ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b.data(), "TensorArithmeticFwd.cpp", operator_name);
                            const Integer *a_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_acutal_shape.data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_acutal_shape.data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_strides.data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_strides.data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));
                                                                                     
                            gpuArithForwardKernel(a.data(), b_dev, c.data(), a_stype, b_stype,
                                                  a_allocated_shape_dev, a_acutal_shape_dev, a_strides_dev, a_dim, a_total,
                                                  b_allocated_shape_dev, b_acutal_shape_dev, b_strides_dev, b_dim, b_total,
                                                  loop_total, arith_type);
                        }   
                        break; 
                    
                    case ndpp_memory::DeviceType::CudaDevice:
                        {
                            const Integer *a_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_allocated_shape.data(), 
                                                                                             "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_acutal_shape.data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_acutal_shape.data(), 
                                                                                          "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *a_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_strides.data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));
                            const Integer *b_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_strides.data(), 
                                                                                     "TensorArithmeticFwd.cpp", operator_name));

                            gpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                                  a_allocated_shape_dev, a_acutal_shape_dev, a_strides_dev, a_dim, a_total,
                                                  b_allocated_shape_dev, b_acutal_shape_dev, b_strides_dev, b_dim, b_total,
                                                  loop_total, arith_type);
                        }
                        break;
                }
            }
            break; 
    }
#else
    // P.S cpu calculation only.
    cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                          a_allocated_shape.data(), a_acutal_shape.data(), a_strides.data(), a_dim, a_total,
                          b_allocated_shape.data(), b_acutal_shape.data(), b_strides.data(), b_dim, b_total,
                          loop_total, arith_type);
#endif
}


void arithmeticForward(const Tensor &a, const Scalar &b, Tensor &c,
                       const Arithmetic arith_type, const std::string &operator_name)
{
    if (!a.data())
    {
        ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                         "The tensor a is empty, and please allocate tensor a first", true);
        exit(EXIT_FAILURE);
    }

    
    const IntArray a_acutal_shape = a.sizes();
    const IntArray a_strides = a.strides();

    const int64_t a_total = static_cast<int64_t>(a.allocations());

    const ndpp_memory::DeviceType a_dtype = a.device();
    const ndpp_memory::ScalarType a_stype = a.scalar();
    const ndpp_memory::ScalarType b_stype = b.type();


    if (!same(a.sizes(), c.sizes()) || a_dtype != c.device() || a_stype != c.scalar())
    {
        c.zerosV(a_acutal_shape, a_strides, a_stype, a_dtype);
    }

    IntArray a_allocated_shape;
    ndpp_data_arch::calcShape(a_acutal_shape, a_strides, a_allocated_shape, a_acutal_shape.device(), 
                              "TensorOperator.cpp", operator_name);

#ifdef CUDA
    switch (a_dtype)
    {
        case ndpp_memory::DeviceType::Host:  
        case ndpp_memory::DeviceType::CudaPinned:
        case ndpp_memory::DeviceType::CudaUnified:
        case ndpp_memory::DeviceType::CudaZeroCpy: 
            {
                cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                      a_allocated_shape.data(), a_acutal_shape.data(), a_strides.data(), a_acutal_shape.size(), a_total,
                                      nullptr, nullptr, nullptr, 0, 1, a_total, arith_type);
            } 
            break;
        
        case ndpp_memory::DeviceType::CudaDevice:  
            {
                const Integer *a_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_allocated_shape.data(), 
                                                                                 "TensorArithmeticFwd.cpp", operator_name));           
                const Integer *a_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_acutal_shape.data(), 
                                                                              "TensorArithmeticFwd.cpp", operator_name));
                const Integer *a_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(a_strides.data(), 
                                                                         "TensorArithmeticFwd.cpp", operator_name));

                gpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                      a_allocated_shape_dev, a_acutal_shape_dev, a_strides_dev, a_acutal_shape.size(), a_total,
                                      nullptr, nullptr, nullptr, 0, 1, a_total, arith_type);
            }
            break; 
    }
#else
    // P.S cpu calculation only.
    cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                          a_allocated_shape.data(), a_acutal_shape.data(), a_strides.data(), a_acutal_shape.size(), a_total,
                          nullptr, nullptr, nullptr, 0, 1, a_total, arith_type);
#endif
}


void arithmeticForward(const Scalar &a, const Tensor &b, Tensor &c,
                       const Arithmetic arith_type, const std::string &operator_name)
{
    if (!b.data())
    {
        ndpp_log::logger("TensorOperator.cpp", operator_name, ndpp_log::RuntimeType::ERROR,
                         "The tensor b is empty, and please allocate tensor b first", true);
        exit(EXIT_FAILURE);
    }

    const IntArray b_acutal_shape = b.sizes();
    const IntArray b_strides = b.strides();

    const int64_t b_total = static_cast<int64_t>(b.allocations());

    const ndpp_memory::DeviceType b_dtype = b.device();
    const ndpp_memory::ScalarType b_stype = b.scalar();
    const ndpp_memory::ScalarType a_stype = a.type();


    if (!same(b.sizes(), c.sizes()) || b_dtype != c.device() || a_stype != c.scalar())
    {
        c.zerosV(b_acutal_shape, b_strides, a_stype, b_dtype);
    }

    IntArray b_allocated_shape;
    ndpp_data_arch::calcShape(b_acutal_shape, b_strides, b_allocated_shape, b_acutal_shape.device(), 
                              "TensorOperator.cpp", operator_name);

#ifdef CUDA
    switch (b_dtype)
    {
        case ndpp_memory::DeviceType::Host:  
        case ndpp_memory::DeviceType::CudaPinned:
        case ndpp_memory::DeviceType::CudaUnified:
        case ndpp_memory::DeviceType::CudaZeroCpy: 
            {
                cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                      nullptr, nullptr, nullptr, 0, 1,
                                      b_allocated_shape.data(), b_acutal_shape.data(), b_strides.data(), b_acutal_shape.size(), b_total,
                                      b_total, arith_type);
            } 
            break;
        
        case ndpp_memory::DeviceType::CudaDevice:  
            {
                const Integer *b_allocated_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_allocated_shape.data(), 
                                                                                 "TensorArithmeticFwd.cpp", operator_name));
                const Integer *b_acutal_shape_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_acutal_shape.data(), 
                                                                              "TensorArithmeticFwd.cpp", operator_name));
                const Integer *b_strides_dev = static_cast<const Integer*>(ndpp_memory::ndpp_cuda::cudaHostGetGpuPointer(b_strides.data(), 
                                                                              "TensorArithmeticFwd.cpp", operator_name));

                gpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                                      nullptr, nullptr, nullptr, 0, 1,
                                      b_allocated_shape_dev, b_acutal_shape_dev, b_strides_dev, b_acutal_shape.size(), b_total,
                                      b_total, arith_type);
            }
            break; 
    }
#else
    // P.S cpu calculation only.
    cpuArithForwardKernel(a.data(), b.data(), c.data(), a_stype, b_stype,
                          nullptr, nullptr, nullptr, 0, 1,
                          b_allocated_shape.data(), b_acutal_shape.data(), b_strides.data(), b_acutal_shape.size(), b_total,
                          b_total, arith_type);
#endif
}

} // namespace ndpp::ndpp_arithmetic

}; // namespace ndpp