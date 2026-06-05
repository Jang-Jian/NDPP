# NDPP

NUMERICAL DATA PIPELINE & PROCESSING.

## ENVIRONMENT.

* Linux (such as Ubuntu).
* gcc & g++.
* nvcc.

## Compliation.

* WITH_CUDA: Supprting the NVIDIA CUDA use (WITH_CUDA=1).
* WITH_HALF: Supprting the float16(half) use (WITH_HALF=1).

```
make -j2 WITH_CUDA=1 WITH_HALF=1
```

## Container.

* ndpp::Array<T>
* ndpp::Tensor