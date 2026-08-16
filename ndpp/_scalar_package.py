from _ctypes_package import scalar_type
from _ndpp_cpython import _cc_scalar


class scalar(_cc_scalar):
    """
    scalar: Representing a 0-dimensional tensor which contains a single element.
    """
    def __init__(self, src = None):
        if src is not None:
            _cc_scalar.__init__(self, src)
        else:
            _cc_scalar.__init__(self)
    
    def data(self) -> float:
        """
        scalar.data: Get the value via float64.
        
        :return: The saved value.
        :rtype: float.
        """
        return self._cc_data()
    
    def type(self) -> scalar_type:
        """
        scalar.type: Get the ndpp.scalar_type.
        
        :return: self scalar type.
        :rtype: ndpp.scalar_type.
        """
        return self._cc_type()