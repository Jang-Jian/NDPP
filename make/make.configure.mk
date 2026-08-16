# Reference: vpath & link headers.
INCLUDE=-I core


# Support: python flag.
ifeq ($(WITH_PYTHON), 1)
PYTHON=1
BUILD_FLAGS+=-DPYTHON
endif


# Support: cuda flag.
ifeq ($(WITH_CUDA), 1)
CUDA=1
BUILD_FLAGS+=-DCUDA
endif


# Support: half flag.
ifeq ($(WITH_HALF), 1)
HALF=1
BUILD_FLAGS+=-DHALF
endif


# Directory: object files (*.o).
ifdef PATH_OBJS
DIR_OBJS=$(PATH_OBJS)/
else
DIR_OBJS=obj/
endif
DIR_KERNEL_OBJS=$(DIR_OBJS)kernel/
DIR_EXE_OBJS=$(DIR_OBJS)exe/
DIR_PYTHON_OBJS=$(DIR_OBJS)python/


# Directory: shared libraries (*.so) for C++.
ifdef PATH_CC_LIBS
DIR_CC_LIBS=$(PATH_LIBS)/
else
DIR_CC_LIBS=lib/
endif


# Directory: shared libraries (*.so) for Python.
DIR_PY_LIBS=ndpp/


INSTALLED_CC_LIBS=/usr/local/ndpp

# Directory: exe files.
ifdef PATH_EXES
DIR_EXES=$(PATH_EXES)/
else
DIR_EXES=exe/
endif


# Directory: python & boost source (PATH_PYTHON_INCLUDE, PATH_PYTHON_LIBS, PATH_BOOST_INCLUDE & PATH_BOOST_LIBS).
ifeq ($(WITH_PYTHON), 1)

PY3_MAIN=$(shell python3 -c "import sys;t='{v[0]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY3_SUB=$(shell python3 -c "import sys;t='{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY3_VER=$(PY3_MAIN).$(PY3_SUB)
PY3_BOOST=$(PY3_MAIN)$(PY3_SUB)

ifeq ($(PY3_VER),$(filter $(PY3_VER), 3.5 3.6 3.7))
PY3_VER:=$(PY3_VER)m
endif

ifdef PATH_PYTHON_INCLUDE
PYTHON_INCLUDE=$(PATH_BOOST_INCLUDE)
else
PYTHON_INCLUDE=/usr/include/python$(PY3_VER)
endif

ifdef PATH_PYTHON_LIBS
PYTHON_LIBS=$(PATH_PYTHON_LIBS)
else
PYTHON_LIBS=/usr/lib/x86_64-linux-gnu
endif

ifdef PATH_BOOST_INCLUDE
BOOST_INCLUDE=$(PATH_BOOST_INCLUDE)
else
BOOST_INCLUDE=/usr/include/boost
endif

ifdef PATH_BOOST_LIBS
BOOST_LIBS=$(PATH_BOOST_LIBS)
else
BOOST_LIBS=/usr/lib/x86_64-linux-gnu
endif

endif


# Directory: cuda source (PATH_CUDA_INCLUDE & PATH_CUDA_LIBS).
ifeq ($(WITH_CUDA), 1)

ifdef PATH_CUDA_INCLUDE
CUDA_INCLUDE=$(PATH_CUDA_INCLUDE)
else
CUDA_INCLUDE=/usr/local/cuda/include
endif

ifdef PATH_CUDA_LIBS
CUDA_LIBS=$(PATH_CUDA_LIBS)
else
CUDA_LIBS=/usr/local/cuda/lib64
endif

endif