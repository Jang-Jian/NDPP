from _ndpp_cpython import _cc_device_status, \
                          _cc_allocation, _cc_reference
class device_status(_cc_device_status):
    pass
allocation = device_status(_cc_allocation)
reference  = device_status(_cc_reference)


from _ndpp_cpython import _cc_scalar_type, \
                          _cc_uint8, _cc_uint16, _cc_uint32, _cc_uint64, \
                          _cc_int8, _cc_int16, _cc_int32, _cc_int64, \
                          _cc_float32, _cc_float64, _cc_bool
class scalar_type(_cc_scalar_type):
    pass
uint8   = scalar_type(_cc_uint8)
uint16  = scalar_type(_cc_uint16)
uint32  = scalar_type(_cc_uint32)
uint64  = scalar_type(_cc_uint64)
int8    = scalar_type(_cc_int8)
int16   = scalar_type(_cc_int16)
int32   = scalar_type(_cc_int32)
int64   = scalar_type(_cc_int64)
try:
    from _ndpp_cpython import _cc_float16
    float16 = scalar_type(_cc_float16)
except Exception as _:
    pass
float32 = scalar_type(_cc_float32)
float64 = scalar_type(_cc_float64)
bool    = scalar_type(_cc_bool)


from _ndpp_cpython import _cc_device_type, _cc_host
class device_type(_cc_device_type):
    pass
host = device_type(_cc_host)
try:
    from _ndpp_cpython import _cc_cuda_device, _cc_cuda_pinned, \
                              _cc_cuda_unified, _cc_cuda_zerocpy
    
    cuda_device  = device_type(_cc_cuda_device)
    cuda_pinned  = device_type(_cc_cuda_pinned)
    cuda_unified = device_type(_cc_cuda_unified)
    cuda_zerocpy = device_type(_cc_cuda_zerocpy)
except Exception as _:
    pass


from _ndpp_cpython import _cc_runtime_type, \
                          _cc_info, _cc_error, _cc_warn, _cc_debug, _cc_end
class runtime_type(_cc_runtime_type):
    pass
info  = runtime_type(_cc_info)
error = runtime_type(_cc_error)
warn  = runtime_type(_cc_warn)
debug = runtime_type(_cc_debug)
end   = runtime_type(_cc_end)


from _ndpp_cpython import _cc_list_push_action, _cc_front, _cc_back
class list_push_action(_cc_list_push_action):
    pass
front = list_push_action(_cc_front)
back  = list_push_action(_cc_back)