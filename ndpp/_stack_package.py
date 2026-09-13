from ndpp_cpython import _cc_stack


class stack(_cc_stack):
    """
    Stack: A LIFO (Last-In, First-Out) / FILO (First-In, Last-Out) data structure.
    """
    def __init__ (self):
        _cc_stack.__init__(self)

    @staticmethod
    def _cc2py(src: _cc_stack) -> "stack":
        """
        _cc2py: Migrates ndpp_cpython._cc_queue to ndpp.stack.
        
        :param src: C++'s PyQueue.
        :type src: ndpp_cpython._cc_stack
        :return: Python's ndpp.stack.
        :rtype: ndpp.stack.
        """
        _dst_pystack = stack()
        _dst_pystack.move_from(src)
        return _dst_pystack

    def size(self) -> int:
        """
        stack.size: Returns the number of elements.

        :return: self dimension.
        :rtype: int.
        """
        return self._cc_size()

    def empty(self) -> bool:
        """
        stack.empty: Checks whether the container adaptor is empty.

        :return: self size whether is empty.
        :rtype: bool.
        """
        return self._cc_empty()

    def pop(self):
        """
        stack.pop: Removes the top element.
        """
        self._cc_pop()

    def clone(self) -> "stack":
        """
        stack.clone: Clone itself to new Queue.

        :return: The new stack.
        :rtype: ndpp.stack.
        """
        return stack._cc2py(self._cc_clone())

    def move_from(self, src: "stack"):
        """
        stack.move_from: Moves all data from outside. \n
        P.S The src will be clean after this call.
        
        :param src: Source of ndpp.stack.
        :type src: ndpp.stack.
        """
        self._cc_move_from(src)

    def top(self) -> object:
        """
        stack.top: Accesses the top element.
        
        :return: The data which saved in top element.
        :rtype: object.
        """
        return self._cc_top()

    def push(self, obj: object):
        """
        stack.push: Pushes the given element value to the top of the stack.
        
        :param obj: The source of object.
        :type obj: object.
        """
        _cc_stack._cc_push(self, obj)