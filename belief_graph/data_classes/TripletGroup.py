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
    span: List[Span]

    @property
    def count(self):
        return len(self.span)

    @staticmethod
    def from_belief_triplet(triplet: BeliefTriplet) -> TripletGroup:
        return TripletGroup(
            head_id=triplet.head_id,
            tail_id=triplet.tail_id,
            relation_ids=triplet.relation_ids,
            confidence=[triplet.confidence],
            span=[triplet.span],
            attr=triplet.attr,
        )

    def __add_triplet(self, triplet: BeliefTriplet):
        self.span.append(triplet.span)

    def __add_tg(self, triplet: TripletGroup):
        self.span += triplet.span
    
    def __add(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if isinstance(triplet, TripletGroup):
            self.__add_tg(triplet)
        self.__add_triplet(triplet)

    def add(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if self == triplet:
            raise ValueError(
                "Cannot add a triplet if it does not have the same head, tail, relation as the triplet group. Use add_if_valid if you only want to add if valid"
            )
        self.__add(triplet)


    def add_if_valid(self, triplet: Union[BeliefTriplet, TripletGroup]):
        if self == triplet:
            self.__add(triplet)
