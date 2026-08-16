# NDPP

NUMERICAL DATA PIPELINE & PROCESSING.

## Environment.

* Linux.

## Compilation.

### Building flags.

Here is all flags you could configure for compilation.

* WITH_CUDA: Supprting the NVIDIA CUDA use (WITH_CUDA=1).
* WITH_HALF: Supprting the float16(half) use (WITH_HALF=1).
* WITH_CUDA_SM: If WITH_CUDA=1 & WITH_HALF=1, it will enable this flag. Please follow the below how to type correct cuda sm number inside.
    * WITH_CUDA_ARCH: Please choose the lowest CUDA SM code from group WITH_CUDA_SM.
    * Rule: If SM{id} or SM_{id}, compute_{id}, you could type WITH_CUDA_SM={id}.
    * Reference: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

* WITH_PYTHON: Supprting the python use (WITH_PYTHON=1).

### Example: building command.

If you wanna build the code for RTX-30 series cards use, you could type

```
make -j2 WITH_CUDA=1 WITH_HALF=1 WITH_CUDA_SM=80,86,87 WITH_CUDA_ARCH=80
```

### Required dependencies.

* gcc & g++.

### Optional dependencies.

### If WITH_CUDA=1.

* nvcc.

### If WITH_PYTHON=1.

* boost.
* python.

## NDPP SDK installation & uninstallation.

* Installed path for C++ libraries: /usr/local/ndpp
* Installed path for Python libraries: /usr/local/lib/python{py-version}/dist-packages

### Basic libraries (C++ only).

If you just install C++ libraries in the host, you could type

```
sudo make install
```

If you wanna remove, you could type

```
sudo make uninstall
```

### All libraries (C++ & Python).

If you wanna install C++ & Python libraries in the host, you could type

```
sudo make install WITH_PYTHON=1
```

If you wanna remove, you could type

```
sudo make uninstall WITH_PYTHON=1
```

## Method.

### Container.

* Tensor (ndpp::Tensor): Representing a multi-dimensional array with a specific scalar type and device.
    * [Sample code for C++](example/cc/tensor_example.cpp).
    * [Sample code for Python](example/python/tensor_example.py).
* Generic (ndpp::Generic): Generic data type implementation using type-erased. 
    * [Sample code for C++](example/cc/generic_example.cpp).
* Array (ndpp::Array): A Container that encapsulates fixed size arrays. 
    * [Sample code for C++](example/cc/array_example.cpp).
* List (ndpp::Tensor): It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    * [Sample code for C++](example/cc/list_example.cpp).
* Stack (ndpp::Stack): A LIFO (Last-In, First-Out) / FILO (First-In, Last-Out) data structure.
    * [Sample code for C++](example/cc/stack_example.cpp).
    * Wrapped via ndpp::List.
* Queue (ndpp::Queue): A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    * [Sample code for C++](example/cc/queue_example.cpp).
    * Wrapped via ndpp::List.

### Algorithm.

* Sorts the elements of the src tensor along a given dimension in ascending order by value (used for ndpp:Tensor).
    * [Sample code for C++](example/cc/sort_example.cpp).