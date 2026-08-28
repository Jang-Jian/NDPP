import os, sys

sys.path.append(os.path.dirname(__file__))

from _ndpp_cpython import __doc__, __version__


from _ctypes_package import *
device_status.__module__    = __name__
scalar_type.__module__      = __name__
device_type.__module__      = __name__
runtime_type.__module__     = __name__
list_push_action.__module__ = __name__


from _cuda_extension import set_cuda_device, \
                            get_cuda_devices, \
                            check_cuda
set_cuda_device.__module__  = __name__
get_cuda_devices.__module__ = __name__
check_cuda.__module__       = __name__


from _logging_package import logger
logger.__module__ = __name__


from _scalar_package import scalar
scalar.__module__ = __name__


from _tensor_pacakge import tensor, zeros, ones, full
tensor.__module__ = __name__
zeros.__module__  = __name__
ones.__module__   = __name__
full.__module__   = __name__


from _list_package import list
list.__module__ = __name__