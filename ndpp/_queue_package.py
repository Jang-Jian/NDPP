from ndpp_cpython import _cc_queue


class queue(_cc_queue):
    """
    queue: Queue: A FIFO (First-In, First-Out) / LILO (Last-In, Last-Out) data structure.
    """
    def __init__(self):
        _cc_queue.__init__(self)

    @staticmethod
    def _cc2py(src: _cc_queue) -> "queue":
        """
        _cc2py: Migrates ndpp_cpython._cc_queue to ndpp.queue.
        
        :param src: C++'s PyQueue.
        :type src: ndpp_cpython._cc_queue
        :return: Python's ndpp.queue.
        :rtype: ndpp.queue.
        """
        _dst_pyqueue = queue()
        _dst_pyqueue.move_from(src)
        return _dst_pyqueue

    def size(self) -> int:
        """
        queue.size: Returns the number of elements.

        :return: self dimension.
        :rtype: int.
        """
        return self._cc_size()

    def empty(self) -> bool:
        """
        queue.empty: Checks whether the container adaptor is empty.

        :return: self size whether is empty.
        :rtype: bool.
        """
        return self._cc_empty()

    def pop(self):
        """
        queue.pop: Removes the top element.
        """
        self._cc_pop()

    def clone(self) -> "queue":
        """
        queue.clone: Clone itself to new Queue.

        :return: The new queue.
        :rtype: ndpp.queue.
        """
        return queue._cc2py(self._cc_clone())

    def move_from(self, src: "queue"):
        """
        queue.move_from: Moves all data from outside. \n
        P.S The src will be clean after this call.
        
        :param src: The source of ndpp.queue.
        :type src: ndpp.queue.
        """
        self._cc_move_from(src)

    def front(self) -> object:
        """
        queue.front: Accesses the first element.
        
        :return: The data which saved in first element.
        :rtype: object.
        """
        return self._cc_front()

    def back(self) -> object:
        """
        queue.back: Accesses the last element.
        
        :return: The data which saved in last element.
        :rtype: object.
        """
        return self._cc_back()

    def push(self, obj: object):
        """
        queue.push: Pushes the given element value to the end of the queue.
        
        :param obj: The source of object.
        :type obj: object.
        """
        _cc_queue._cc_push(self, obj)