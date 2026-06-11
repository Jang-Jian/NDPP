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

## Function.

### Container.

* ndpp::Array<T>: [Sample code for C++](core/example/array_example.cpp).
* ndpp::Tensor: [Sample code for C++](core/example/tensor_example.cpp).

### Algorithm.

* ndpp::sort: [Sample code for C++](core/example/sort_example.cpp).