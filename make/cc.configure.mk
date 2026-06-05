include make/make.configure.mk


CFLAGS=-std=c++11 -fPIC -O3 -funroll-loops -march=native -fopenmp -Wall

ifdef BUILD_FLAGS
CC=g++ $(BUILD_FLAGS) $(INCLUDE) $(CFLAGS)
else
CC=g++ $(INCLUDE) $(CFLAGS)
endif

ARFLAGS=rcs
AR=ar $(ARFLAGS)