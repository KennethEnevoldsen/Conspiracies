from spacy.tokens import Span
from typing import List
from .BeliefTriplet import BeliefTriplet
from __future__ import annotations


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
