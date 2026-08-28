from _ndpp_cpython import _cc_list
from _ctypes_package import list_push_action


class list(_cc_list):
    """
    list: It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    """
    def __init__(self,):
        _cc_list.__init__(self)

    @staticmethod
    def _cc2py(src: _cc_list) -> "list":
        """
        list._cc2py: Migrates _ndpp_cpython._cc_list to ndpp.list.

        :type src: _ndpp_cpython._cc_list
        :return: Python's ndpp.list.
        :rtype: ndpp.list.
        """
        _dst_pylist = list()
        _dst_pylist.migrate(src)
        return _dst_pylist

    def size(self) -> int:
        """
        Docstring for size

        :return: Description
        :rtype: int
        """
        return self._cc_size()
    
    def clear(self):
        """
        Docstring for clear
        """
        self._cc_clear()

    def erase(self, position: int):
        """
        Docstring for erase
        
        :param position: Description
        """
        self._cc_erase(position)

    def clone(self) -> "list":
        """
        Docstring for clone
        
        :param self: Description
        :return: Description
        :rtype: list
        """
        return list._cc2py(self._cc_clone())
                                               
    def migrate(self, src: "list"):
        """
        Docstring for migrate

        :param src: Description
        :type src: "list"
        """
        self._cc_migrate(src)

    def pop_front(self):
        """
        Docstring for pop_front.
        """
        self._cc_pop_front()

    def pop_back(self):
        """
        Docstring for pop_back.
        """
        self._cc_pop_back()
    
    def front(self) -> object:
        """
        Docstring for front
        
        :param self: Description
        :return: Description
        :rtype: object
        """
        return self._cc_front()
    
    def back(self) -> object:
        """
        Docstring for back
        
        :return: Description
        :rtype: object
        """
        return self._cc_back()

    def insert(self, position: int, obj: object, action: list_push_action):
        """
        Docstring for insert

        :param position: Description
        :type position: int
        :param obj: Description
        :type obj: object
        :param action: Description
        :type action: list_push_action
        """
        _cc_list._cc_insert(self, position, obj, action)

    def reverse(self):
        """
        Docstring for reverse
        """
        self._cc_reverse()

    def splice(self, position: int, src: "list"):
        """
        Docstring for splice
        
        :param self: Description
        :param position: Description
        :type position: int
        :param src: Description
        :type src: "list"
        """
        self._cc_splice(position, src)

    def push_back(self, obj: object):
        """
        Docstring for push_back
        
        :param obj: Description
        :type obj: object
        """
        _cc_list._cc_push_back(self, obj)

    def push_front(self, obj: object):
        """
        Docstring for push_front
        
        :param self: Description
        :param obj: Description
        :type obj: object
        """
        _cc_list._cc_push_front(self, obj)