# https://stackoverflow.com/questions/9446634/makefile-vpath-not-working-for-header-files
VPATH=core/src/logging:core/src/array:core/src/base:core/src/tensor:core/src/algorithm:core/src/list

# kernel.
KERNEL_OBJS=Logging.o GenericBase.o ArrayStype.o ScalarType.o DataArch.o TensorBase.o Tensor.o Sort.o ListBase.o List.o ListHelper.o
ifeq ($(WITH_CUDA), 1)
KERNEL_OBJS+=GpuDataProc.o
endif

# example.
EXEAMPLE_OBJS=array_example.o generic_example.o scalar_example.o tensor_example.o sort_example.o list_example.o

# libraries.
CC_SHARED=$(DIR_LIBS)libndpp.so
CC_STATIC=$(DIR_LIBS)libndpp.a
PY_SHARED=$(DIR_LIBS)ndpp.so