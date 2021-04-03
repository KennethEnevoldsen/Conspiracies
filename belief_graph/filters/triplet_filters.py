"""
this contain local filters for belief triplets
"""
from functools import partial
from typing import Callable, Generator, Iterable, List, Optional, Union

from spacy.tokens import Span, Token

from ..data_classes import BeliefTriplet
from ..utils import is_a_range as is_cont


def make_simple_triplet_filters(
    validators_heads: List[Union[str, Callable]] = [valid_pos, valid_entities],
    validators_tails: List[Union[str, Callable]] = [valid_pos, valid_entities],
    validators_relations: List[Union[str, Callable]] = [valid_pos],
    reject_entire: bool = False,
    continous: bool = True,
    confidence_threshold: Optional[float] = None,
) -> List[Callable]:
    """
    reject_entire (bool): Only applied to relation. Should the entire triplet be rejected if one tokens does not pass the filter or should the
    remainder of the relation tokens constitute a relation.
    continuous (bool): Should the relation be continuous.
    """
    funcs = []
    if confidence_threshold is not None:
        f = partial(filter_confidence, confidence_threshold)
        funcs.append(f)
    for func in validators_heads:
        f = partial(filter_head, func=func)
        funcs.append(f)
    for func in validators_tails:
        f = partial(filter_tail, func=func)
        funcs.append(f)
    for func in validators_relations:
        f = partial(filter_relations, func=func, reject_entire=reject_entire)
        funcs.append(f)
    if continous:
        funcs.append(filter_is_continuous)


### --- BeliefTriplet -> None/BeliefTriplet --- ###


def filter_triplet_span(
    self, triplet: BeliefTriplet, attr: str, func: Callable
) -> Union[BeliefTriplet, None]:
    if func(getattr(triplet, attr)):
        return triplet
    return None


def filter_head(
    self, triplet: BeliefTriplet, func: Callable
) -> Union[BeliefTriplet, None]:
    return self.filter_span(triplet, attr="head_span", func=func)


def filter_tail(
    self, triplet: BeliefTriplet, func: Callable
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

    triplet._relation_ids = tuple(i for i, k in zip(triplet._relation_ids, keep) if k)
    if triplet._relation_ids:
        return triplet


def filter_is_continuous(triplet: BeliefTriplet) -> Union[BeliefTriplet, None]:
    if is_cont(triplet._relation_ids):
        return triplet


def filter_confidence(triplet: BeliefTriplet, threshold) -> Union[BeliefTriplet, None]:
    if triplet.confidence < threshold:
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
