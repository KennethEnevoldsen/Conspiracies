"""
Filters applied to triplet groups
"""
from functools import partial
from typing import Callable, List, Union

from ..data_classes import TripletGroup


def count_filter(tg: TripletGroup, count: int) -> Union[TripletGroup, None]:
    """
    count (int): The minimum required count
    """
    if count <= tg.count:
        return tg
