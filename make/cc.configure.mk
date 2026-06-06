include make/make.configure.mk


CCSTD=-std=c++11
CCFLAGS= -fPIC -O3 -funroll-loops -march=native -Wall
CC_DEPENDENCY+=-fopenmp

ifdef BUILD_FLAGS
CC=g++ $(BUILD_FLAGS) $(INCLUDE) $(CCSTD) $(CCFLAGS)
else
CC=g++ $(INCLUDE) $(CCSTD) $(CCFLAGS)
endif

ARFLAGS=rcs
AR=ar $(ARFLAGS)