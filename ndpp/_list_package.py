from ndpp_cpython import _cc_list
from ndpp_cpython import node, list_iterator
from _ctypes_package import list_push_action


class list(_cc_list):
    """
    list: It is a container that supports inserting generic data type, and it's implemented as a doubly-linked list.
    """
    def __init__(self):
        _cc_list.__init__(self)

    @staticmethod
    def _cc2py(src: _cc_list) -> "list":
        """
        list._cc2py: Migrates ndpp_cpython._cc_list to ndpp.list.

        :type src: ndpp_cpython._cc_list
        :return: Python's ndpp.list.
        :rtype: ndpp.list.
        """
        _dst_pylist = list()
        _dst_pylist.migrate(src)
        return _dst_pylist

    def __iter__ (self) -> list_iterator:
        return _cc_list._cc_iter(self)

    def size(self) -> int:
        """
        list.size: Returns the number of elements.

        :return: self dimension.
        :rtype: int.
        """
        return self._cc_size()
    
    def clear(self):
        """
        list.clear: Clears the contents.
        """
        self._cc_clear()

    def erase(self, position: int):
        """
        list.erase: Erases element via specified location..
        
        :param position: The index of position.
        """
        self._cc_erase(position)

    def clone(self) -> "list":
        """
        list.clone: Clone itself to new List.
        
        :param self: Description
        :return: Description
        :rtype: list
        """
        return list._cc2py(self._cc_clone())
                                               
    def migrate(self, src: "list"):
        """
        list.clone: List migration. \n
        P.S The src will be clean after this call.

        :param src: Source of ndpp.list.
        :type src: ndpp.list.
        """
        self._cc_migrate(src)

    def pop_front(self):
        """
        list.pop_front: Removes the first element.
        """
        self._cc_pop_front()

    def pop_back(self):
        """
        list.pop_back: Removes the last element.
        """
        self._cc_pop_back()
    
    def front(self) -> object:
        """
        list.front: Returns a reference to the first element in the container.

        :return: The data which saved in first element.
        :rtype: object.
        """
        return self._cc_front()
    
    def back(self) -> object:
        """
        list.back: Returns a reference to the last element in the container.
        
        :return: The data which saved in last element.
        :rtype: object.
        """
        return self._cc_back()

    def insert(self, position: int, obj: object, action: list_push_action):
        """
        list.insert: Insert the node in front/rear of specified location.

        :param position: The index of position.
        :type position: int.
        :param obj: The source of object.
        :type obj: object.
        :param action: The action for pushing new data.
        :type action: ndpp.list_push_action.
        """
        _cc_list._cc_insert(self, position, obj, action)

    def reverse(self):
        """
        list.reverse: Reverses the order of the elements in the container.
        """
        self._cc_reverse()

    def splice(self, position: int, src: "list"):
        """
        list.splice: Moves elements from another List. \n
        P.S The src will be clean after this call.
        
        :param position: The index of position.
        :type position: int.
        :param src: The source of ndpp.list.
        :type src: ndpp.list.
        """
        self._cc_splice(position, src)

    def push_back(self, obj: object):
        """
        list.push_back: Adds an element to the end.
        
        :param obj: The source of object.
        :type obj: object.
        """
        _cc_list._cc_push_back(self, obj)

    def push_front(self, obj: object):
        """
        list.push_front: Inserts an element to the beginning.
        
        :param obj: The source of object.
        :type obj: object.
        """
        _cc_list._cc_push_front(self, obj)