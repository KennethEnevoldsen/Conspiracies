from __future__ import annotations

from typing import List, Union

from pydantic.main import UNTOUCHED_TYPES
from spacy.tokens import Span

from .BeliefTriplet import BeliefTriplet


class TripletGroup(BeliefTriplet):
    """
    this is a triplet group for collapsing
    span is a list consisting of spacy spans or None
    """

    confidence: List[float]
    spans: List[Span]

    @property
    def count(self):
        return len(self.spans)

    @staticmethod
    def from_belief_triplet(triplet: BeliefTriplet) -> TripletGroup:
        return TripletGroup(
            head_id=triplet.head_id,
            tail_id=triplet.tail_id,
            relation_ids=triplet.relation_ids,
            confidence=[triplet.confidence],
            span=triplet.span,
            spans=[triplet.span],
            getter=triplet.getter,
        )

    def add_triplet(self, triplet: BeliefTriplet):
        self.spans.append(triplet.span)

    def add_tg(self, triplet: TripletGroup):
        self.spans += triplet.spans

    def __add(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if isinstance(triplet, TripletGroup):
            self.add_tg(triplet)
        self.add_triplet(triplet)

    def add(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if self == triplet:
            self.__add(triplet)
        else:
            raise ValueError(
                "Cannot add a triplet if it does not have the same head, tail, relation as the triplet group. Use add_if_valid if you only want to add if valid"
            )

    def add_if_valid(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if self == triplet:
            self.__add(triplet)

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [
                ("head", self.head),
                ("relation", self.relation),
                ("tail", self.tail),
                ("count", self.count),
            ]
        )
