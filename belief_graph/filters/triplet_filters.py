"""
this contain local filters for belief triplets
"""
from functools import partial
from typing import Callable, Generator, Iterable, List, Optional, Union

from spacy.tokens import Span, Token

from ..data_classes import BeliefTriplet
from .utils import is_cont_integer as is_cont

### --- BeliefTriplet -> None/BeliefTriplet --- ###


def filter_triplet_span(
    triplet: BeliefTriplet, attr: str, func: Callable
) -> Union[BeliefTriplet, None]:
    if func(getattr(triplet, attr)):
        return triplet
    return None


def filter_head(triplet: BeliefTriplet, func: Callable) -> Union[BeliefTriplet, None]:
    return filter_triplet_span(triplet, attr="head_span", func=func)


def filter_tail(triplet: BeliefTriplet, func: Callable) -> Union[BeliefTriplet, None]:
    return filter_triplet_span(triplet, attr="tail_span", func=func)


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

    triplet.relation_ids = tuple(i for i, k in zip(triplet.relation_ids, keep) if k)
    if triplet.relation_ids:
        return triplet


def filter_is_continuous(triplet: BeliefTriplet) -> Union[BeliefTriplet, None]:
    if is_cont(triplet.relation_ids):
        return triplet


def filter_confidence(triplet: BeliefTriplet, threshold) -> Union[BeliefTriplet, None]:
    if triplet.confidence >= threshold:
        return triplet


### --- Span -> Bool --- ###


def valid_attribute_span(
    span: Span,
    attr: str,
    valid: Optional[set] = None,
    invalid: Optional[set] = None,
):
    return all(valid_attribute_token(t, attr, valid, invalid) for t in span)


def valid_entities(
    span: Span,
    valid: Optional[set] = {"LOC", "PER", "ORG"},
    invalid: Optional[set] = None,
):
    return valid_attribute_span(span, "ent_type_", valid, invalid)


def valid_pos(
    span: Span,
    valid: Optional[set] = None,
    invalid: Optional[set] = {"PUNCT", "SPACE", "NUM"},
):
    return valid_attribute_span(span, "pos_", valid, invalid)


def valid_dependency(
    span: Span,
    valid: Optional[set] = None,
    invalid: Optional[set] = None,
):
    return valid_attribute_span(span, "dep_", valid, invalid)


### --- Token -> Bool --- ###


def valid_attribute_token(
    token: Token,
    attr: str,
    valid: Optional[set] = None,
    invalid: Optional[set] = None,
):
    att = getattr(token, attr)
    return ((valid is None) or (att in valid)) and (
        (invalid is None) or (att not in invalid)
    )
