from belief_graph.data_classes.BeliefTriplet import BeliefTriplet
from typing import Callable
from .triplet_filters import (
    filter_tails,
    filter_heads,
    valid_entities,
    valid_pos,
    filter_relations,
)
from functools import partial

PREDEFINED_TF = {"punct": pre_filter_pos(), "ent": pre_filter_ent()}
PREDEFINED_GF = {"count": pre_filter_count()}
PREDEFINED_LF = dict()
PREDEFINED_LF = {
    "triplet_filters": PREDEFINED_TF,
    "group_filters": PREDEFINED_GF,
    "list filters": PREDEFINED_LF,
}

def pre_filter_pos(valid_pos: Callable = valid_pos):
    valid_pos = [valid_pos]
    ft = partial(filter_tails, funs = valid_pos)
    fh = partial(filter_heads, funs = valid_pos)
    fr = partial(filter_tails, funs = valid_pos)
    def wrapper(triplet: BeliefTriplet):
        return fr(ft(fh(triplet)))
    return wrapper

def pre_filter_ent(valid_entities: Callable=valid_entities):
    valid_entities = [valid_entities]
    ft = partial(filter_tails, funs = valid_entities)
    fh = partial(filter_heads, funs = valid_entities)

    def wrapper(triplet: BeliefTriplet):
        return ft(fh(triplet))
    return wrapper