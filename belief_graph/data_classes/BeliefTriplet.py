from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from .BeliefTriplet import BeliefTriplet


class BaseTriplet(BaseModel):
    """
    The base class for other belief triplets.
    """

    class Config:
        arbitrary_types_allowed = True

    head: str
    tail: str
    relation: str
    confidence: Optional[float] = None
    sentence: str

    def __eq__(self, other: BaseTriplet) -> bool:
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
                ("span", self.sentence),
            ]
        )

    def __lt__(self, other: BaseTriplet):
        return self.head < other.head

    def __gt__(self, other: BaseTriplet):
        return self.head > other.head


def text_getter(span: Span):
    return span.text


class BeliefTriplet(BaseTriplet):
    class Config:
        arbitrary_types_allowed = True

    head: Optional[str] = None
    tail: Optional[str] = None
    relation: Optional[str] = None
    sentence: Optional[str] = None
    head_id: int
    tail_id: int
    relation_ids: Tuple[int, ...]
    confidence: float
    getter: Callable = text_getter
    is_offloaded: bool = False
    _span_slice: Optional[slice] = None
    _doc_reference: Optional[str] = None
    __span: Optional[Span] = None
    __vocab: Optional[Vocab] = None

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
    def span(self) -> Span:
        if self.__span:
            return self.__span
        doc = Doc(self.__vocab).from_disk(self.doc_reference)
        self.__span = doc[self.span_slice]
        return self.__span

    @staticmethod
    def from_parse(
        self,
        head_id: int,
        tail_id: int,
        relation_ids: Tuple[int, ...],
        confidence: float,
        span: Span,
    ):

        bt = BeliefTriplet(
            head_id=head_id,
            tail_id=tail_id,
            relation_ids=relation_ids,
            confidence=confidence,
            span=span,
        )
        bt.update()
        return bt

    def set_getter(self, getter: Callable):
        self.getter = getter
        self.update()

    def get_tail(self) -> str:
        return self.getter(self.tail_span)

    def get_head(self) -> str:
        return self.getter(self.head_span)

    def get_relation(self) -> str:
        return " ".join(self.getter(t) for t in self.relation_list)

    def get_sentence(self) -> str:
        return self.span.text

    def update(self):
        self.sentence = self.get_sentence()
        self.head = self.get_head()
        self.tail = self.get_tail()
        self.relation = self.get_relation()

    def clean(self):
        self.__span = None

    def offload(
        self,
        vocab: Optional[Vocab] = None,
        id_set: Optional[set] = None,
        dir="triplet_docs",
    ):
        if self.is_offloaded:
            return None

        if vocab is not None:
            self.vocab = vocab
        if self.vocab is None:
            raise ValueError(
                "No vocab is specified. Writing a doc without a Vocab will lead to the doc not being properly recoverable"
            )

        span = self.__span
        path = spacy_doc_to_dir(span.doc, id_set=id_set, dir=dir)

        self._span_slice = slice(span.start, span.end)
        self._doc_reference = path

        self.clean()


def spacy_doc_to_dir(doc, id_set: Optional[set] = None, dir="triplet_docs") -> str:
    Path(dir).mkdir(parents=True, exist_ok=True)

    if id_set is None:
        id_set = set(f[:-4] for f in os.listdir(dir) if f.endswith(".doc"))

    id_ = id(doc)
    path = os.path.join(dir, id_ + ".doc")
    if id_ not in id_set:
        doc.to_disk(path)

    return path
