import ndpp
import numpy as np


def _print_tensor(src):
    src_shape = src.sizes()
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            for k in range(src_shape[2]):
                print(float(b[i][j][k]), ",", end="")
            print()
        print()
    print()


if __name__ == "__main__":

    if ndpp.check_cuda():
        ndpp.set_cuda_device(0)
        device_t = ndpp.cuda_device
    else:
        device_t = ndpp.host

    #a = np.zeros((5,6,8))
    a = ndpp.full(5, (5,6,8), ndpp.float32, device_t)
    print(a)
    b = ndpp.tensor()
    #b.from_numpy(a, ndpp.host, ndpp.host, ndpp.reference)
    b.from_tensor(a, ndpp.host, ndpp.reference)
    b_shape = b.sizes()


    for i in range(b_shape[0]):
        for j in range(b_shape[1]):
            for k in range(b_shape[2]):
                b[i][j][k] = float(k + j * b_shape[2] + i * b_shape[1] * b_shape[2])

    b += 2
    b -= 2
    b = 2.0 * b 
    b = b / 2 

    if ndpp.check_cuda():
        print("=========")
        c = b.to(ndpp.int32, ndpp.cuda_device)
        c = c.to(ndpp.int64, ndpp.cuda_pinned)
        print( type(c))
        _print_tensor(c)
        print(c.sizes())
        print(c.strides())
        print(c.device())
        print(c.scalar())
        print("=========")

    for i in range(b_shape[0]):
        for j in range(b_shape[1]):
            for k in range(b_shape[2]):
                print(float(b[i][j][k]), ",", end="")
            print()
        print()
    print()

    #print(a)
    _print_tensor(a)

    print(b.sizes())
    print(b.strides())
    print(b.status())
    print(b.scalar())

    c = ndpp.tensor()
    c.from_tensor(b, ndpp.host, ndpp.reference)
    print(c.sizes())
    print(c.strides())
    print(c.status())


    c_copy = c.clone(ndpp.host)
    print(c_copy.sizes())
    print(c_copy.strides())
    print(c_copy.status())