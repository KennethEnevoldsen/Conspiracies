def is_a_range(L):
    """
    checks if a list is equal to a range

    Examples:
    >>> is_a_range([2, 3, 4])
    True
    >>> is_a_range([2, 4, 5])
    False
    >>> is_a_range(L=[1, 3, 2])
    False
    """
    L_ = range(L[0], L[-1] + 1)
    if len(L_) != len(L):
        return False
    for i, j in zip(L_, L):
        if i != j:
            return False
    return True
