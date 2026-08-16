from _scalar_package import scalar
from _ctypes_package import scalar_type, device_type, device_status, float32, host
from _ndpp_cpython import _cc_tensor, _cc_none, _cc_reversed, \
                          _cc_zeros, _cc_ones, _cc_full


class tensor(_cc_tensor):
    """
    tensor: Representing a multi-dimensional array with a specific scalar type and device.
    """
    def __init__(self,):
        _cc_tensor.__init__(self)

    @staticmethod
    def _cc2py(src: _cc_tensor) -> "tensor":
        """
        _cc2py: Migrates _ndpp_cpython._cc_tensor to ndpp.tensor.
        
        :param src: C++'s PyTensor.
        :type src: _ndpp_cpython._cc_tensor
        :return: Python's tensor.
        :rtype: ndpp.tensor.
        """
        _dst_pytensor = tensor()
        _dst_pytensor.migrate(src)
        return _dst_pytensor
    
    def __getitem__(self, index: int) -> "tensor":
        return tensor._cc2py(self._cc_getitem(index))

    def __setitem__(self, index: int, rhs: "numbers.Number"):
       self._cc_setitem(index, rhs)

    def __add__(self, b: any) -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_add_tensor(self, b, _cc_none))
    
    def __sub__(self, b: any) -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_sub_tensor(self, b, _cc_none))
    
    def __mul__(self, b: any) -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_mul_tensor(self, b, _cc_none))
    
    def __truediv__(self, b: any) -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_div_tensor(self, b, _cc_none))
    
    def __floordiv__(self, b: any) -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_floordiv_tensor(self, b, _cc_none))
    
    def __radd__(self, b: "numbers.Number") -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_add_tensor(self, b, _cc_reversed))
    
    def __rsub__(self, b: "numbers.Number") -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_sub_tensor(self, b, _cc_reversed))
    
    def __rmul__(self, b: "numbers.Number") -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_mul_tensor(self, b, _cc_reversed))
    
    def __rtruediv__(self, b: "numbers.Number") -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_div_tensor(self, b, _cc_reversed))
    
    def __rfloordiv__(self, b: "numbers.Number") -> "tensor":
        return tensor._cc2py(_cc_tensor._cc_floordiv_tensor(self, b, _cc_reversed))
    
    def __iadd__(self, b: any) -> "tensor":
        _cc_tensor._cc_iadd_tensor(self, b)
        return self
    
    def __isub__(self, b: any) -> "tensor":
        _cc_tensor._cc_isub_tensor(self, b)
        return self

    def __imul__(self, b: any) -> "tensor":
        _cc_tensor._cc_imul_tensor(self, b)
        return self

    def __itruediv__(self, b: any) -> "tensor":
        _cc_tensor._cc_idiv_tensor(self, b)
        return self

    def dim(self) -> int:
        """
        tensor.dim: Get the dimension.

        :return: self dimension.
        :rtype: int.
        """
        return self._cc_dim()
    
    def elements(self) -> int:
        """
        tensor.elements: Get the element size on acutal use. \n
        P.S tensor.elements() isn't necessarily same as tensor.allocations().

        :return: self element size.
        :rtype: int.
        """
        return self._cc_elements()
    
    def allocations(self) -> int:
        """
        tensor.allocations: Get the actual size on memory allocation. \n
        P.S allocations() isn't necessarily same as elements().

        :return: self allocated size.
        :rtype: int.
        """
        return self._cc_allocations()
    
    def scalar(self) -> scalar_type:
        """
        tensor.scalar: Get the ndpp.scalar_type (data type).

        :return: self scalar type.
        :rtype: ndpp.scalar_type.
        """
        return scalar_type(self._cc_scalar())
    
    def device(self) -> device_type:
        """
        tensor.device: Get the ndpp.device_type (memory type).

        :return: self device type.
        :rtype: ndpp.device_type.
        """
        return device_type(self._cc_device())
    
    def status(self) -> device_status:
        """
        tensor.status: Get the ndpp.device_status (allocation status).

        :return: self device status.
        :rtype: ndpp.device_status.
        """
        return device_status(self._cc_status())

    def sizes(self) -> tuple:
        """
        tensor.sizes: Get the shape.

        :return: self shape.
        :rtype: tuple.
        """
        return self._cc_sizes()
    
    def strides(self) -> tuple:
        """
        tensor.strides: Get the strides.

        :return: self strides.
        :rtype: tuple.
        """
        return self._cc_strides()
    
    def destory(self):
        """
        tensor.destory: Deallocate the memory.
        """
        self._cc_destory()
    
    def clone(self, dtype: device_type) -> "tensor":
        """
        tensor.clone: Clone itself to new Tensor via ndpp.device_type.

        :param dtype: Destination of device type.
        :type dtype: ndpp.device_type.
        :return: The new tensor.
        :rtype: ndpp.tensor.
        """
        return tensor._cc2py(self._cc_clone(dtype))
    
    def to(self, stype: scalar_type, dtype: device_type) -> "tensor":
        """
        tensor.to: Convert self to new Tensor with ndpp.scalar_type & ndpp.device_type.
        
        :param stype: Destination of scalar type
        :type stype: ndpp.scalar_type.
        :param dtype: Destination of device type
        :type dtype: ndpp.device_type.
        :return: The new tensor.
        :rtype: ndpp.tensor.
        """
        return tensor._cc2py(self._cc_to(stype, dtype))
    
    def migrate(self, src: "tensor"):
        """
        tensor.migrate: tensor migration. \n
        P.S The src will be clean after this call.

        :param src: Source of ndpp.tensor.
        :type src: ndpp.tensor.
        """
        self._cc_migrate(src)

    def from_tensor(self, src: "tensor",
                          dst_dtype: device_type,
                          copy_status: device_status):
        """
        tensor.from_tensor: Copy data from ndpp.tensor. \n
        P.S If the copy_status is ndpp.reference, the dst_dtype doesn't work.
        
        :param src: Source of ndpp.tensor.
        :type src: ndpp.tensor.
        :param dst_dtype: Destination of device type.
        :type dst_dtype: ndpp.device_type.
        :param copy_status: Copied status for this work.
        :type copy_status: ndpp.device_status
        """
        self._cc_from_tensor(src, dst_dtype, copy_status)

    def from_numpy(self, src: "numpy.ndarray",
                         src_dtype: device_type,
                         dst_dtype: device_type,
                         copy_status: device_status):
        """
        tensor.from_python: Copy data from numpy.ndarray. \n
        P.S If the copy_status is ndpp.reference, the dst_dtype doesn't work.
        
        :param src: Source of numpy.ndarray
        :type src: numpy.ndarray.
        :param src_dtype: Source(src) of device type.
        :type src_dtype: ndpp.device_type.
        :param dst_dtype: Destination of device type.
        :type dst_dtype: ndpp.device_type.
        :param copy_status: Copied status for this work.
        :type copy_status: ndpp.device_status.
        """
        self._cc_from_numpy(src, src_dtype, dst_dtype, copy_status)


def zeros(size: tuple, scalar_t: scalar_type = float32, device_t: device_type = host) -> tensor:
    """
    zeros: Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument.
    
    :param size: Destination of shape.
    :type size: tuple.
    :param scalar_t: Destination of scalar type (default: ndpp.float32).
    :type scalar_t: ndpp.scalar_type.
    :param device_t: Destination of device type (default: ndpp.host).
    :type device_t: ndpp.device_type.
    :return: The new tensor.
    :rtype: ndpp.tensor
    """
    return tensor._cc2py(_cc_zeros(size, scalar_t, device_t))


def ones(size: tuple, scalar_t: scalar_type = float32, device_t: device_type = host) -> tensor:
    """
    onces:  Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument.
    
    :param size: Destination of shape.
    :type size: tuple.
    :param scalar_t: Destination of scalar type (default: ndpp.float32).
    :type scalar_t: ndpp.scalar_type.
    :param device_t: Destination of device type (default: ndpp.host).
    :type device_t: ndpp.device_type.
    :return: The new tensor.
    :rtype: ndpp.tensor.
    """
    return tensor._cc2py(_cc_ones(size, scalar_t, device_t))


def full(fill_value: "numbers.Number", size: tuple, scalar_t: scalar_type = float32, device_t: device_type = host) -> tensor:
    """
    full: Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument.
    
    :param fill_value: The filled scalar.
    :type fill_value: numbers.Number.
    :param size: Destination of shape.
    :type size: tuple.
    :param scalar_t: Destination of scalar type (default: ndpp.float32).
    :type scalar_t: ndpp.scalar_type.
    :param device_t: Destination of device type (default: ndpp.host).
    :type device_t: ndpp.device_type.
    :return: The new tensor.
    :rtype: ndpp.tensor.
    """
    return tensor._cc2py(_cc_full(scalar(fill_value), size, scalar_t, device_t))