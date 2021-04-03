"""
this contain local filters for belief triplets
"""
from functools import partial
from typing import Callable, Generator, Iterable, List, Union

from spacy.tokens import Token

from ..data_classes import BeliefTriplet
from ..utils import is_a_range as is_cont



### Iterable[BeliefTriplet] -> Generator[BeliefTriplet] filters ###

def filter_heads(
    triplets: Iterable[BeliefTriplet], funs: Iterable[Callable]
) -> Generator[BeliefTriplet]:
    """
    triplets (Iterable[BeliefTriplet]): a list of triplets which heads should be filtered
    funs (Iterable[Callable]): a list of functions which should take a spacy token as input and return a boolean.
    """
    for triplet in triplets:
        if all(f(triplet.head_token) for f in funs):
            yield triplet


def filter_tails(
    triplets: Iterable[BeliefTriplet], funs: Union[Iterable[Callable]]
) -> Generator[BeliefTriplet]:
    """
    triplets (Iterable[BeliefTriplet]): a list of triplets which tails should be filtered
    funs (Iterable[Callable]): a list of functions which should take a spacy token as input and return a boolean.
    """
    for triplet in triplets:
        if all(f(triplet.tail_token) for f in funs):
            yield triplet


def filter_relations(
    triplets: Iterable[BeliefTriplet],
    funs: Iterable[Callable],
    reject_entire: bool = False,
    continuous: bool = False,
) -> Generator[BeliefTriplet]:
    """
    triplets (Iterable[BeliefTriplet]): a list of triplets which heads should be filtered
    funs (Iterable[Callable]): a list of functions which should take a spacy token as input and return a boolean.
    reject_entire (bool): Should the entire triplet be rejected if one token does not pass the filter or should the remainder of the relation tokens constitute a relation.
    continuous (bool): Should the relation be continuous. if
    """
    for triplet in triplets:
        reject_triplet = False
        keep = []
        for i, t in enumerate(triplet.relation_list):
            passed = all(f(t) for f in funs)
            if not passed:
                if reject_entire:
                    reject_triplet = True
                    break
                keep.append(False)
            else:
                keep.append(True)

        if reject_triplet:
            continue
        elif passed:
            yield triplet
        else:
            if continuous is True and not is_cont(triplet._relation_ids):
                continue

            triplet._relation_ids = tuple(
                i for i, k in zip(triplet._relation_ids, keep) if k
            )
            yield triplet


### Token -> Bool ###

def valid_attribute(
    token: Token,
    attr: str,
    allowed: set = set(),
    disallowed: set = set(),
):
    att = getattr(token, attr)
    return (att in allowed) and (att not in disallowed)


def valid_entities(
    token: Token,
    allowed: set = {"LOC", "PER", "ORG"},
    disallowed: set = set(),
):
    return valid_attribute(token, "ent_type_", allowed, disallowed)


def valid_pos(
    token: Token,
    allowed: set = set(),
    disallowed: set = {"PUNCT", "SPACE", "NUM"},
):
    return valid_attribute(token, "pos_", allowed, disallowed)


def valid_dependency(
    token: Token,
    allowed: set = set(),
    disallowed: set = set(),
):
    return valid_attribute(token, "dep_", allowed, disallowed)
