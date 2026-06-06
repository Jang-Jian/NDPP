include make/cc.configure.mk
include make/make.configure.mk
include make/cuda.configure.mk
include make/python.configure.mk
include make/binaries.configure.mk


BUILD_FLAGS := $(sort $(BUILD_FLAGS))
CC_DEPENDENCY := $(sort $(CC_DEPENDENCY))
KERNELS := $(addprefix $(DIR_KERNEL_OBJS), $(KERNEL_OBJS))
EXEAMPLE_OBJS := $(addprefix $(DIR_EXE_OBJS), $(EXEAMPLE_OBJS))
EXEAMPLES := $(addprefix $(DIR_EXES), $(patsubst %.o,%,$(notdir $(EXEAMPLE_OBJS))))


all: obj $(CC_STATIC) $(CC_SHARED) $(EXEAMPLE_OBJS) $(EXEAMPLES)

obj:
	mkdir -p $(DIR_OBJS) $(DIR_KERNEL_OBJS) $(DIR_EXE_OBJS)
	mkdir -p $(DIR_LIBS)
	mkdir -p $(DIR_EXES)

clean:
	rm -rf $(DIR_OBJS)
	rm -rf $(DIR_LIBS) 
	rm -rf $(DIR_EXES) 


$(CC_STATIC): $(KERNELS)
	$(AR) $@ $^


$(CC_SHARED): $(KERNELS) 
	$(CC) -shared $(KERNELS) $(CC_DEPENDENCY) -o $(CC_SHARED)

$(DIR_EXES)%: $(DIR_EXE_OBJS)%.o $(CC_STATIC)
	$(CC) $^ $(CC_DEPENDENCY) -o $@ 

$(DIR_EXE_OBJS)%.o: core/example/%.cpp
	$(CC) $(CC_DEPENDENCY) -c $< -o $@

$(DIR_KERNEL_OBJS)%.o: %.cpp
	$(CC) $(CC_DEPENDENCY) -c $< -o $@

$(DIR_KERNEL_OBJS)%.o: %.cu
	$(NVCC) $(CUDA_DEPENDENCY) -c $< -o $@