import ndpp
#import numpy as np


# def foo():
#     _a = ndpp.list()
#     _a.insert(0, np.zeros((6, 6)), ndpp.front)
#     bb = None
#     for h in _a:
#         bb = h._cc_data()
#         print(bb)
#     return bb


if __name__ == "__main__":
    a = ndpp.list()

    #aaa = foo()
    #print(aaa)
   

    print("a before:", a.size())
    a.insert(0, "888", ndpp.front)
    print("a front:", a.front(), "type:", type(a.front()), "data:", a.front().data())
    print("a front:", a.front(), "type:", type(a.front()), "data:", a.front().exists())

    print("a after:", a.size())
    a.insert(1, 999, ndpp.back)
    print("a after:", a.size())
    a.insert(2, "abd", ndpp.back)
    print("a after:", a.size())

    a.push_back({'test': 123})

    a.reverse()

    print("iterator-1:", next(iter(a)))

    # loopping via python iterator.
    print("loopping via python iterator.")
    for _a in a:
        print(_a.data()) 
    print()

    # loopping via c-style node->next.
    print("loopping via c-style node->next.")
    node = a.front()
    while node.exists():
        print(node.data())
        node = node.next_node()

    print()
    #a.erase(2)

    # data access.
    print("a front:", a.front(), "type:", type(a.front()), "data:", a.front().data())


    # C style doublely-linked list for accessing next & prev node.
    print("a->next:", a.front(), "type:", type(a.front()), "data:", a.front().next_node().data())
    print("a->prev:", a.front(), "type:", type(a.front()), "data:", a.front().prev_node().data())
    print("a->next->next:", a.front(), "type:", type(a.front()), "data:", a.front().next_node().next_node().data())

    print("a back:", a.back(), "type:", type(a.back()), "data:", a.back().data())

    b = a.clone()
    print("b list:", type(b))
    
    print("b front:", b.front(), "type:", type(b.front()))
    print("b back:", b.back(), "type:", type(b.back()))

    a.clear()
    #b.clear()

    print("a after:", a.size())
    print("b after:", b.size())