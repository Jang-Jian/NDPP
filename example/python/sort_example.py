import ndpp


def _print_tensor(src):
    src_shape = src.sizes()
    for i in range(src_shape[0]):
        for j in range(src_shape[1]):
            for k in range(src_shape[2]):
                print(float(src[i][j][k]), ",", end="")
            print()
        print()
    print()


if __name__ == "__main__":
    tensor_test = ndpp.zeros((3, 4, 5), ndpp.float32, ndpp.host)

    shape = tensor_test.sizes()


    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                tensor_test[i][j][k] = float(k + j * shape[2] + i * shape[1] * shape[2])

    
    _print_tensor(tensor_test)

    tensor_dst = ndpp.sort(tensor_test, descending=True)

    _print_tensor(tensor_dst)