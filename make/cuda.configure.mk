include make/make.configure.mk


ifeq ($(WITH_CUDA), 1)
CUDA_SETTINGS=-I$(CUDA_INCLUDE) -L$(CUDA_LIBS) -lcuda -lcudart -lcublas -lcurand

CC_DEPENDENCY+=$(CUDA_SETTINGS)
endif