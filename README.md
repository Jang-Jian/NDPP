# NDPP

NUMERICAL DATA PIPELINE & PROCESSING.

## Environment.

* Linux.

## Compilation.

### Building flags.

Here is all flags you could configure for compilation.

* WITH_CUDA: Uses `WITH_CUDA=1` for supporting cuda.
* WITH_HALF: Uses `WITH_HALF=1` for supporting the float16(half).
* WITH_CUDA_SM: If WITH_CUDA=1 & WITH_HALF=1, it will enable this flag. Please follow the below how to type correct cuda sm number inside.
    * WITH_CUDA_ARCH: Please choose the lowest CUDA SM code from group `WITH_CUDA_SM`.
    * Rule: If SM{id} or SM_{id}, compute_{id}, you could type `WITH_CUDA_SM={id}`.
    * Reference: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

* WITH_PYTHON: Uses `WITH_PYTHON=1` for supporting the python.

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

* python.
* boost.python.

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

All examples that we save in [Samples for C++](example/cc) & [Samples for Python](example/python) respectively.

### Container.

* Tensor: Representing a multi-dimensional array with a specific scalar type and device.
    * For C++.
        * [All API interface-1](core/include/tensor/Tensor.hpp).
        * [All API interface-2](core/include/tensor/TensorHelper.hpp).
        * [The sample code](example/cc/tensor_example.cpp).
    * For Python.
        * [All API interface of ndpp.tensor](ndpp/_tensor_pacakge.py).
        * [The sample code](example/python/tensor_example.py).
* Generic: Generic data type implementation using type-erased. 
    * For C++.
        * [ALL API interface](core/include/base/GenericBase.hpp).
        * [The sample code](example/cc/generic_example.cpp).
* Array: A Container that encapsulates fixed size arrays. 
    * For C++.
        * [All API interface](core/include/array/Array.hpp).
        * [The sample code](example/cc/array_example.cpp).
* List: It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    * For C++.
        * [All API interface](core/include/list/List.hpp).
        * [The sample code](example/cc/list_example.cpp).
    * For Python.
        * [All API interface](ndpp/_list_pacakge.py).
        * [The sample code](example/python/list_example.py).
* Stack: A LIFO (Last-In, First-Out) / FILO (First-In, Last-Out) data structure.
    * For C++.
        * [All API interface](core/include/segment/Stack.hpp).
        * [The sample code](example/cc/stack_example.cpp).
    * For Python.
        * [All API interface](ndpp/_stack_package.py).
        * [The sample code](example/python/stack_example.py).
* Queue: A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    * For C++.
        * [All API interface](core/include/segment/Queue.hpp).
        * [The sample code](example/cc/queue_example.cpp).
    * For Python.
        * [All API interface](ndpp/_queue_package.py).
        * [The sample code](example/python/queue_example.py).

### Algorithm.

* Sort: Sorts the elements of the src tensor along a given dimension in ascending order by value (used for ndpp:Tensor).
    * For C++.
        * [The API interface](core/include/algorithm/Sort.hpp).
        * [The sample code](example/cc/sort_example.cpp).
    * For Python.
        * [The API interface](ndpp/_algorithm_package.py)
        * [The sample code](example/python/sort_example.py).