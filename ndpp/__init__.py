import os, sys

sys.path.append(os.path.dirname(__file__))

from ndpp_cpython import __doc__, __version__


from _ctypes_package import device_status, scalar_type, device_type, \
                            runtime_type, list_push_action
device_status.__module__    = __name__
scalar_type.__module__      = __name__
device_type.__module__      = __name__
runtime_type.__module__     = __name__
list_push_action.__module__ = __name__

from _ctypes_package import allocation, reference
from _ctypes_package import uint8, uint16, uint32, uint64, \
                            int8, int16, int32, int64, \
                            float32, float64, bool
try:
    from _ctypes_package import float16
except Exception as _:
    pass                        

from _ctypes_package import host
try:
    from _ctypes_package import cuda_device, cuda_pinned, \
                                cuda_unified, cuda_zerocpy
except Exception as _:
    pass

from _ctypes_package import info, error, warn, debug, end
from _ctypes_package import front, back


from _cuda_extension import set_cuda_device,  \
                            get_cuda_devices, \
                            check_cuda
set_cuda_device.__module__  = __name__
get_cuda_devices.__module__ = __name__
check_cuda.__module__       = __name__


from _logging_package import logger
logger.__module__ = __name__


from _scalar_package import scalar
scalar.__module__ = __name__


from _tensor_package import tensor, zeros, ones, full
tensor.__module__ = __name__
zeros.__module__  = __name__
ones.__module__   = __name__
full.__module__   = __name__


from _list_package import list, node, list_iterator
list.__module__          = __name__
node.__module__          = __name__
list_iterator.__module__ = __name__


from _queue_package import queue
queue.__module__ = __name__


from _stack_package import stack
stack.__module__ = __name__


from _algorithm_package import sort
sort.__module__   = __name__