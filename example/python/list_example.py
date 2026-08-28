import ndpp
#import numpy as np


# def foo():
#     _a = ndpp.list()
#     _a.insert(0, np.zeros((6, 6)), ndpp.front)
#     return _a.front()


if __name__ == "__main__":
    a = ndpp.list()

    #aaa = foo()
    #print(aaa)

    print("a before:", a.size())
    a.insert(0, "888", ndpp.front)
    print("a after:", a.size())
    a.insert(1, 999, ndpp.back)
    print("a after:", a.size())

    a.push_back({'test': 123})

    a.reverse()

    a.erase(2)

    print("a front:", a.front(), "type:", type(a.front()))
    print("a back:", a.back(), "type:", type(a.back()))

    b = a.clone()
    print("b list:", type(b))
    
    print("b front:", b.front(), "type:", type(b.front()))
    print("b back:", b.back(), "type:", type(b.back()))

    a.clear()
    #b.clear()

    print("a after:", a.size())
    print("b after:", b.size())