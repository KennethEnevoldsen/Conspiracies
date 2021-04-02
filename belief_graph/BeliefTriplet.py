"""
This Script contains the Belief Triplet class
"""
from typing import Tuple
from pydantic import BaseModel
from spacy.tokens import Span


class BeliefTriplet(BaseModel):
    """
    the data class for the belief triplet. Which contains besides
    """

    class Config:
        arbitrary_types_allowed = True

    path: Tuple[int, ...]
    confidence: float
    span: Span

    @property
    def _head_id(self):
        return self.path[0]

    @property
    def _relation_ids(self):
        return self.path[1:-1]

    @property
    def _tail_id(self):
        return self.path[-1]

    @property
    def tail(self):
        return self.tail_token.text

    @property
    def head(self):
        return self.head_token.text

    @property
    def tail_token(self):
        span = self.span
        return span._.nctokens[self._tail_id]

    @property
    def head_token(self):
        span = self.span
        return span._.nctokens[self._head_id]

    @property
    def relation_list(self):
        span = self.span
        return [span._.nctokens[i] for i in self._relation_ids]

    @property
    def relation(self):
        return " ".join(t.text for t in self.relation_list)

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
