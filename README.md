# NDPP

NUMERICAL DATA PIPELINE & PROCESSING.

## Environment.

* Linux.
* gcc & g++.
* nvcc.

## Compilation.

### Building flags.

Here is all flags you could configure for compilation.

* WITH_CUDA: Supprting the NVIDIA CUDA use (WITH_CUDA=1).
* WITH_HALF: Supprting the float16(half) use (WITH_HALF=1).
* WITH_CUDA_SM: If WITH_HALF=1, it will enable this flag. Please follow the below how to type correct cuda sm number inside.
    * WITH_CUDA_ARCH: Please choose the lowest CUDA SM code from group WITH_CUDA_SM.
    * Rule: If SM{id} or SM_{id}, compute_{id}, you could type WITH_CUDA_SM={id}.
    * Reference: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

### Example: building command.

If you wanna build the code for RTX-30 series cards use, you could type

```
make -j2 WITH_CUDA=1 WITH_HALF=1 WITH_CUDA_SM=80,86,87 WITH_CUDA_ARCH=80
```

## Method.

### Container.

* Generic (ndpp::Generic): Generic data type implementation using type-erased. 
    * [Sample code for C++](core/example/generic_example.cpp).
* Array (ndpp::Array): A Container that encapsulates fixed size arrays. 
    * [Sample code for C++](core/example/array_example.cpp).
* Tensor (ndpp::Tensor): Representing a multi-dimensional array with a specific scalar type and device.
    * [Sample code for C++](core/example/tensor_example.cpp).
* List (ndpp::Tensor): It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    * [Sample code for C++](core/example/list_example.cpp).
* Stack (ndpp::Stack): A LIFO (Last-In, First-Out) / FILO (First-In, Last-Out) data structure.
    * [Sample code for C++](core/example/stack_example.cpp).
    * The lazy object: Wrapped via ndpp::List.
* Queue (ndpp::Queue): A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    * [Sample code for C++](core/example/queue_example.cpp).
    * The lazy object: Wrapped via ndpp::List.

### Algorithm.

* Sorts the elements of the src tensor along a given dimension in ascending order by value (used for ndpp:Tensor).
    * [Sample code for C++](core/example/sort_example.cpp).