from typing import Iterable


def is_a_range(L: Iterable[int]) -> bool:
    """
    checks if a list is equal to a range

    Examples:
    >>> is_a_range([2, 3, 4])
    True
    >>> is_a_range([2, 4, 5])
    False
    >>> is_a_range(L=[1, 3, 2])
    False
    >>> is_a_range(L=[3, 2, 1])
    False
    """
    if len(L) == 1:
        return True
    L_ = range(L[0], L[-1] + 1)
    if len(L_) != len(L):
        return False
    for i, j in zip(L_, L):
        if i != j:
            return False
    return True

def is_cont_integer(L: Iterable[int]) -> bool:
    """
    checks if list is an continous integer

    Examples:
    >>> is_a_range([2, 3, 4])
    True
    >>> is_a_range([2, 4, 5])
    False
    >>> is_a_range(L=[1, 3, 2])
    False
    >>> is_a_range(L=[3, 2, 1])
    True

    """
    return is_a_range(L) or is_a_range(L[::-1])