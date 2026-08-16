#ifdef CUDA
#include <include/extension/CudaProc.hpp>
#endif


namespace ndpp
{

namespace ndpp_python
{

#ifdef CUDA
inline void setCudaDevice(const int device_t)
{
    ndpp_memory::ndpp_cuda::cudaExecDevice(device_t, "PyExtension.hpp", "ndpp::ndpp_python::setCudaDevice()");
}

inline int getCudaDevices()
{
    return ndpp_memory::ndpp_cuda::cudaGetDevices("PyExtension.hpp", "ndpp::ndpp_python::getCudaDevices()");
}
#endif


inline bool checkCuda()
{
#ifdef CUDA
    return ndpp_memory::ndpp_cuda::cudaAvailable("PyExtension.hpp", "ndpp::ndpp_python::checkCuda()");
#else
    return false;
#endif
}

}; // namespace ndpp::ndpp_python

}; // namespace ndpp