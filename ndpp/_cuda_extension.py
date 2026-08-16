from sys import exit
from _ctypes_package import error
from _logging_package import logger


def set_cuda_device(device_t: int):
    """
    set_cuda_device: Configure the cuda device id for execution.
    
    :param device_t: Cuda device id.
    :type device_t: int.
    """
    try:
        from _ndpp_cpython import _cc_set_cuda_device
        _cc_set_cuda_device(device_t)
    except Exception as _set_cuda_err:
        logger("ndpp", "set_cuda_device", error, str(_set_cuda_err) + ", it's probably that this ndpp didn't build with CUDA.", True)
        exit(0)


def get_cuda_devices() -> int:
    """
    get_cuda_devices: Get number of cuda devices.
    
    :return: Number of cuda devices.
    :rtype: int.
    """
    try:
        from _ndpp_cpython import _cc_get_cuda_devices
        return _cc_get_cuda_devices()
    except Exception as _set_cuda_err:
        logger("ndpp", "set_cuda_device", error, str(_set_cuda_err) + ", it's probably that this ndpp didn't build with CUDA.", True)
        exit(0) 

def check_cuda() -> bool:
        
    """
    check_cuda: Check whether cuda is available.
    
    :return: whether cuda is available or not.
    :rtype: bool.
    """
    from _ndpp_cpython import _cc_check_cuda
    return _cc_check_cuda()