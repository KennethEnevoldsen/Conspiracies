"""
Filters applied to triplet groups
"""
from functools import partial
from typing import Callable, List, Union

from ..data_classes import TripletGroup


def make_simple_group_filters(count: int) -> List[Callable]:
    funcs = []
    if count:
        f = partial(count_filter, count)
        funcs.append(f)
    return funcs


def count_filter(tg: TripletGroup, count: int) -> Union[TripletGroup, None]:
    """
    count (int): The minimum required count
    """
    if count <= tg.count:
        return tg
