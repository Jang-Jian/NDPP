from _tensor_package import tensor


def sort(src: tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> tensor:
    """
    sort: Sorts the elements of the src tensor along a given dimension in ascending order by value.
    """
    return tensor._cc2py(_cc_sort(src, dim, descending, stable))