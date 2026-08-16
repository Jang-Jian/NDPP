include make/make.configure.mk


# https://stackoverflow.com/questions/9446634/makefile-vpath-not-working-for-header-files
VPATH=core/src/logging:core/src/array:core/src/base:core/src/tensor:core/src/algorithm:core/src/list:core/src/python

CC_EXAMPLE=example/cc

# cc kernel.
KERNEL_OBJS=Logging.o GenericBase.o ArrayStype.o ScalarType.o DataArch.o DataProc.o TensorBase.o Tensor.o TensorArithmeticFwd.o Sort.o ListBase.o List.o ListHelper.o
ifeq ($(WITH_CUDA), 1)
KERNEL_OBJS+=GpuDataProc.o GpuTensorArithmeticFwd.o
endif


# regular libraries.
CC_SHARED=$(DIR_CC_LIBS)libndpp.so
CC_STATIC=$(DIR_CC_LIBS)libndpp.a


ifeq ($(WITH_PYTHON), 1)
# python kernel.
PYLIB_OBJS+=_ndpp_cpython.o
PYPACK_OBJS+=PyScalar.o PyTensor.o 

# python libraries.
CC_PY_SHARED=$(DIR_CC_LIBS)libndpp_python.so
CC_PY_STATIC=$(DIR_CC_LIBS)libndpp_python.a
PY_SHARED=$(DIR_PY_LIBS)_ndpp_cpython.so
endif


# example.
EXEAMPLE_OBJS=array_example.o generic_example.o scalar_example.o tensor_example.o sort_example.o list_example.o stack_example.o queue_example.o