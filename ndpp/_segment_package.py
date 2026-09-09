from ndpp_cpython import _cc_queue


class queue(_cc_queue):
    """
    queue: Queue: A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    """
    def __init__(self):
        _cc_queue.__init__(self)

    def size(self) -> int:
        """
        Docstring for size

        :return: Description
        :rtype: int
        """
        return self._cc_size()

    def empty(self) -> bool:
        """
        Docstring for empty

        :return: Description
        :rtype: int
        """
        return self._cc_empty()

    def pop(self):
        """
        Docstring for pop
        """
        self._cc_pop()

    def clone(self) -> "queue":
        """
        Docstring for clone
        
        :param self: Description
        :return: Description
        :rtype: queue
        """
        self._cc_clone()

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
        
        :param self: Description
        :return: Description
        :rtype: object
        """
        return self._cc_back()

    def push(self, obj: object):
        """
        Docstring for push
        
        :param obj: Description
        :type obj: object
        """
        _cc_queue._cc_push(self, obj)