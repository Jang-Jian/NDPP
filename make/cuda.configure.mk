include make/cc.configure.mk
include make/make.configure.mk

SHELL := bash


ifeq ($(WITH_CUDA), 1)

CUDA_DEPENDENCY=-I$(CUDA_INCLUDE) -L$(CUDA_LIBS) -lcuda -lcudart -lcublas -lcurand
CC_DEPENDENCY+=$(CUDA_DEPENDENCY)

# Detect the nvcc version.
NVCC_VER = $(shell nvcc --version | grep release | cut -d, -f2 | cut -c10-15)


# Generate the CUDA sm codes.
ifeq ($(WITH_HALF), 1)

ifneq ($(and $(WITH_CUDA_SM),$(WITH_CUDA_ARCH)),)

SM_COMMA= ,
SM_EMPTY= 
SM_SPACE= $(SM_EMPTY) $(SM_EMPTY)
SM_LIST=$(subst $(SM_COMMA),$(SM_SPACE),$(WITH_CUDA_SM))

$(foreach SM_CODE,$(SM_LIST),\
	$(eval code_flag=-gencode arch=compute_$(SM_CODE),code=sm_$(SM_CODE)) \
	$(eval CUDA_SM+=$(code_flag)) \
)

CUDA_ARCH=-arch=sm_$(WITH_CUDA_ARCH) $(CUDA_SM)

endif
endif


ifdef CUDA_ARCH
NVCC = nvcc $(BUILD_FLAGS) $(INCLUDE) $(CCSTD) $(CUDA_ARCH) --use_fast_math --compiler-options "$(CCFLAGS)" -Xcompiler -fopenmp
else
NVCC = nvcc $(BUILD_FLAGS) $(INCLUDE) $(CCSTD) --use_fast_math --compiler-options "$(CCFLAGS)" -Xcompiler -fopenmp
endif

endif