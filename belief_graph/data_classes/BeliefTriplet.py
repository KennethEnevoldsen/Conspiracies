from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab


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

    def isin(
        self,
        values: Union[set, Iterable],
        apply_to: List[str] = ["head", "tail", "relation"],
    ) -> bool:
        """
        Tests if triplet attributes is in values.

        apply_to a list of strings. Valid strings include "head", "tail", "relation", "sentence"
        """
        return (
            ("head" in apply_to and self.head in values)
            or ("tail" in apply_to and self.tail in values)
            or ("relation" in apply_to and self.relation in values)
            or ("sentence" in apply_to and self.sentence in values)
        )

    def head_isin(self, values: Union[set, Iterable]) -> bool:
        return self.head in values

    def tail_isin(self, values: Union[set, Iterable]) -> bool:
        return self.tail in values

    def relation_isin(self, values: Union[set, Iterable]) -> bool:
        return self.tail in values

    def sentence_isin(self, values: Union[set, Iterable]) -> bool:
        return self.sentence in values

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
                ("sentence", self.sentence),
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
    span_slice: Optional[slice] = None
    doc_path: Optional[str] = None
    span_reference: Optional[Span] = None
    vocab: Optional[Vocab] = None

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
        if self.span_reference:
            return self.span_reference
        doc = Doc(self.vocab).from_disk(self.doc_path)
        self.span_reference = doc[self.span_slice]
        return self.span_reference

    @staticmethod
    def from_parse(
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
            span_reference=span,
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
        self.span_reference = None

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
        span = self.span_reference
        if self.vocab is None:
            self.vocab = span.vocab

        path = spacy_doc_to_dir(span.doc, id_set=id_set, dir=dir)

        self.span_slice = slice(span.start, span.end)
        self.doc_path = path

        self.clean()


def spacy_doc_to_dir(doc, id_set: Optional[set] = None, dir="triplet_docs") -> str:
    Path(dir).mkdir(parents=True, exist_ok=True)

    if id_set is None:
        id_set = set(int(f[:-4]) for f in os.listdir(dir) if f.endswith(".doc"))

    id_ = id(doc)
    path = os.path.join(dir, str(id_) + ".doc")
    if id_ not in id_set:
        doc.to_disk(path)

    return path
