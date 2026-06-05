# https://stackoverflow.com/questions/9446634/makefile-vpath-not-working-for-header-files
VPATH=core/src/logging:core/src/array:core/src/base:core/src/tensor

# kernel.
KERNEL_OBJS=Logging.o ArrayStype.o ScalarType.o DataArch.o TensorBase.o Tensor.o

# example.
EXEAMPLE_OBJS=array_example.o scalar_example.o tensor_example.o

# libraries.
CC_SHARED=$(DIR_LIBS)libndpp.so
CC_STATIC=$(DIR_LIBS)libndpp.a
PY_SHARED=$(DIR_LIBS)ndpp.so