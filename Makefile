include make/cc.configure.mk
include make/make.configure.mk
include make/cuda.configure.mk
include make/python.configure.mk
include make/binaries.configure.mk


BUILD_FLAGS := $(sort $(BUILD_FLAGS))
CC_DEPENDENCY := $(sort $(CC_DEPENDENCY))
CU_DEPENDENCY := $(sort $(CU_DEPENDENCY))
PY_DEPENDENCY := $(sort $(PY_DEPENDENCY))
CC_KERNELS := $(addprefix $(DIR_KERNEL_OBJS), $(KERNEL_OBJS))
PY_KERNELS := $(addprefix $(DIR_PYTHON_OBJS), $(PYPACK_OBJS))
PYLIB_OBJS := $(addprefix $(DIR_PYTHON_OBJS), $(PYLIB_OBJS))
EXEAMPLE_OBJS := $(addprefix $(DIR_EXE_OBJS), $(EXEAMPLE_OBJS))
EXEAMPLES := $(addprefix $(DIR_EXES), $(patsubst %.o,%,$(notdir $(EXEAMPLE_OBJS))))
NDPP_IN_PATH := $(shell whereis $(INSTALLED_CC_LIBS))
#SHELL := /bin/bash

define CHECK_LIB_EXISTS
	ifeq ($$(wildcard $(1)),)
		$(info $(1) doesn't exist. Please build this library or check your building-command first.)
		$(2) := 0
	else
		$(2) := 1
	endif
endef


all: obj $(CC_STATIC) $(CC_SHARED) $(EXEAMPLE_OBJS) $(EXEAMPLES) $(CC_PY_STATIC) $(CC_PY_SHARED) $(PY_SHARED)

obj:
	mkdir -p $(DIR_OBJS) $(DIR_KERNEL_OBJS) $(DIR_EXE_OBJS) $(DIR_PYTHON_OBJS)
	mkdir -p $(DIR_CC_LIBS)
	mkdir -p $(DIR_EXES)

install: #CC_SHARED_EXIST:=$(call CHECK_LIB_EXISTS,$(CC_SHARED))
	$(eval $(call CHECK_LIB_EXISTS,$(CC_STATIC),CC_STATIC_EXIST))
	$(eval $(call CHECK_LIB_EXISTS,$(CC_SHARED),CC_SHARED_EXIST))
	@echo "Value of CC_STATIC_EXIST is: $(CC_STATIC_EXIST)"
	@echo "Value of CC_SHARED_EXIST is: $(CC_SHARED_EXIST)"

ifeq ($(CC_STATIC_EXIST)$(CC_SHARED_EXIST),1)
   	mkdir -p $(INSTALLED_CC_LIBS)
	cp -r core/include $(INSTALLED_CC_LIBS)
	cp -r $(DIR_CC_LIBS) $(INSTALLED_CC_LIBS)
endif

ifeq ($(NDPP_IN_PATH), ndpp:) # If there is no ndpp's global path (saved in /etc/profile). it will be added automatically.
	echo 'export PATH=$(PATH):$(INSTALLED_CC_LIBS)' | tee -a /etc/profile
	source /etc/profile
endif
ifeq ($(WITH_PYTHON), 1)
ifeq ("$(wildcard $(CC_PY_STATIC))","")
	$(info $(CC_PY_STATIC) doesn't exist. Please build this library or check your building-command first.)
endif
ifeq ("$(wildcard $(CC_PY_SHARED))","")
	$(info $(CC_PY_SHARED) doesn't exist. Please build this library or check your building-command first.)
endif
ifeq ("$(wildcard $(PY_SHARED))","")
	$(info $(PY_SHARED) doesn't exist. Please build this library or check your building-command first.)
endif
	cp -r $(DIR_PY_LIBS) /usr/local/lib/python$(PY3_VER)/dist-packages
endif

uninstall:
	rm -rf $(INSTALLED_CC_LIBS)
ifneq ($(NDPP_IN_PATH), ndpp:)
	sed -i '\#$(INSTALLED_CC_LIBS)#d' /etc/profile
	source /etc/profile
endif
ifeq ($(WITH_PYTHON), 1)
	rm -rf /usr/local/lib/python$(PY3_VER)/dist-packages/$(DIR_PY_LIBS)
endif

clean:
	rm -rf $(DIR_OBJS)
	rm -rf $(DIR_CC_LIBS) $(DIR_PY_LIBS)*.so $(DIR_PY_LIBS)__pycache__
	rm -rf $(DIR_EXES) 


$(CC_STATIC): $(CC_KERNELS)
	$(AR) $@ $^

$(CC_SHARED): $(CC_KERNELS) 
	$(CC) -shared $(CC_KERNELS) $(CC_DEPENDENCY) -o $(CC_SHARED)

$(CC_PY_STATIC): $(CC_KERNELS) $(PY_KERNELS)
	$(AR) $@ $^

$(CC_PY_SHARED): $(CC_KERNELS) $(PY_KERNELS)
	$(CC) -shared $(CC_KERNELS) $(PY_KERNELS) $(CC_DEPENDENCY) $(PY_DEPENDENCY) -o $(CC_PY_SHARED)

$(PY_SHARED): $(CC_KERNELS) $(PY_KERNELS) $(PYLIB_OBJS)
	$(CC) -shared $(CC_KERNELS) $(PY_KERNELS) $(PYLIB_OBJS) $(CC_DEPENDENCY) $(PY_DEPENDENCY) -o $(PY_SHARED)


$(DIR_EXES)%: $(DIR_EXE_OBJS)%.o $(CC_STATIC)
	$(CC) $^ $(CC_DEPENDENCY) -o $@ 

$(DIR_EXE_OBJS)%.o: $(CC_EXAMPLE)/%.cpp
	$(CC) $(CC_DEPENDENCY) -c $< -o $@

$(DIR_KERNEL_OBJS)%.o: %.cpp
	$(CC) $(CC_DEPENDENCY) -c $< -o $@

$(DIR_KERNEL_OBJS)%.o: %.cu
	$(NVCC) $(CU_DEPENDENCY) -c $< -o $@

$(DIR_PYTHON_OBJS)%.o: %.cpp
	$(CC) $(CC_DEPENDENCY) $(PY_DEPENDENCY) -c $< -o $@