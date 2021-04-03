"""
this contain local filters for belief triplets
"""
from functools import partial
from typing import Callable, Generator, Iterable, List, Union

from spacy.tokens import Span, Token

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

### --- BeliefTriplet -> None/BeliefTriplet --- ###

def filter_triplet_span(
    self, triplet: BeliefTriplet, attr: str, func: Callable
) -> Union[BeliefTriplet, None]:
    if func(getattr(triplet, attr)):
        return triplet
    return None

def filter_head(
    self, triplet: BeliefTriplet, attr: str, func: Callable
) -> Union[BeliefTriplet, None]:
    return self.filter_span(triplet, attr="head_span", func=func)

def filter_tails(
    self, triplet: BeliefTriplet, attr: str, func: Callable
) -> Union[BeliefTriplet, None]:
    return self.filter_span(triplet, attr="tail_span", func=func)


def filter_relations(
    triplet: BeliefTriplet,
    func: Callable, 
    reject_entire: bool = False,
) -> Union[BeliefTriplet, None]:
    keep = []
    for i, span in enumerate(triplet.relation_list):
        keep.append(func(span))
        if not keep[-1] and reject_entire:
                return None
    
    triplet._relation_ids = tuple(
        i for i, k in zip(triplet._relation_ids, keep) if k
    )
    if triplet._relation_ids:
        return triplet

### --- Span -> Bool --- ###

def valid_attribute_span(
    span: Span,
    attr: str,
    allowed: set = set(),
    disallowed: set = set(),
):
    return all(valid_attribute_token(t, attr, allowed, disallowed) for t in span)


def valid_entities(
    span: Span,
    allowed: set = {"LOC", "PER", "ORG"},
    disallowed: set = set(),
):
    return valid_attribute_span(span, "ent_type_", allowed, disallowed)


def valid_pos(
    span: Span,
    allowed: set = set(),
    disallowed: set = {"PUNCT", "SPACE", "NUM"},
):
    return valid_attribute_span(span, "pos_", allowed, disallowed)


def valid_dependency(
    span: Span,
    allowed: set = set(),
    disallowed: set = set(),
):
    return valid_attribute_span(span, "dep_", allowed, disallowed)



### --- Token -> Bool --- ###


def valid_attribute_token(
    token: Token,
    attr: str,
    allowed: set = set(),
    disallowed: set = set(),
):
    att = getattr(token, attr)
    return (att in allowed) and (att not in disallowed)



