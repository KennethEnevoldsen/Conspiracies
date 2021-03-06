"""
This Script contains the Belief Triplet class
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel
from spacy.tokens import Span



def text_getter(span: Span):
    return span.text


class BeliefTriplet(BaseModel):
    """
    the data class for the belief triplet. Which contains besides

    ht_getter ("lemma"|"text"): should it return the full head tail (text) or just the lemma (lemma).
    """

    class Config:
        arbitrary_types_allowed = True

    head_id: int
    tail_id: int
    relation_ids: Tuple[int, ...]
    confidence: float
    span: Optional[Span] = None
    getter: Callable = text_getter

    @property
    def tail(self) -> str:
        return self.getter(self.tail_span)

    @property
    def head(self) -> str:
        return self.getter(self.head_span)

    @property
    def tail_span(self) -> Span:
        span = self.span
        return span._.nctokens[self.tail_id]

    @property
    def head_span(self) -> Span:
        span = self.span
        return span._.nctokens[self.head_id]

    @property
    def relation_list(self) -> List[Span]:
        span = self.span
        return [span._.nctokens[i] for i in self.relation_ids]

    @property
    def relation(self) -> str:
        return " ".join(self.getter(t) for t in self.relation_list)

    def __eq__(self, other: BeliefTriplet) -> bool:
        if (
            (self.head == other.head)
            and (self.tail == other.tail)
            and (self.relation == other.relation)
        ):
            return True
        return False

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [
                ("head", self.head),
                ("relation", self.relation),
                ("tail", self.tail),
                ("confidence", round(self.confidence, 2)),
                ("span", self.span),
            ]
        )

    def __lt__(self, other: BeliefTriplet):
        return self.head < other.head

    def __gt__(self, other: BeliefTriplet):
        return self.head > other.head



