from __future__ import annotations

from typing import List, Optional, Union

from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from .BeliefTriplet import BaseTriplet, BeliefTriplet, spacy_doc_to_dir


class TripletGroup(BaseTriplet):
    """
    this is a triplet group for collapsing
    span is a list consisting of spacy spans or None
    """

    class Config:
        arbitrary_types_allowed = True

    confidence: List[float]
    sentence: List[str]
    confidence: List[float]
    vocab: Optional[Vocab] = None
    span_slice: List[slice] = []
    doc_path: List[str] = []
    span_reference: List[Union[None, Span]] = []

    @property
    def count(self):
        return len(self.sentence)

    @staticmethod
    def from_belief_triplet(triplet: BaseTriplet, offload=False) -> TripletGroup:
        tg = TripletGroup(
            head=triplet.head,
            tail=triplet.tail,
            relation=triplet.relation,
            sentence=[triplet.sentence],
            confidence=[triplet.confidence],
        )
        if not isinstance(triplet, BeliefTriplet):
            return tg
        if offload is True:
            triplet.offload()

        tg.span_slice.append(triplet.span_slice)
        tg.doc_path.append(triplet.doc_path)

        if triplet.is_offloaded is True:  # if it is offloaded
            tg.vocab = triplet.vocab
            tg.span_reference.append(None)
        else:
            tg.span_reference.append(triplet.span)
        return tg

    @property
    def is_offloaded(self) -> bool:
        return all(s is None for s in self.span_reference)

    def offload(
        self,
        vocab: Optional[Vocab] = None,
        id_set: Optional[set] = None,
        dir="triplet_docs",
    ):
        if vocab is not None:
            self.vocab = vocab
        i = 0
        while self.vocab is None:
            span = self.span_reference[i]
            self.vocab = span.vocab if span is not None else None
            i += 1

        for i, s in enumerate(self.span_reference):
            if s is None:
                continue
            path = spacy_doc_to_dir(s.doc, id_set=id_set, dir=dir)
            self.span_slice[i] = slice(s.start, s.end)
            self.doc_path[i] = path
        self.clean()

    def clean(self):
        self.span_reference = [None] * self.count

    @property
    def span(self):
        for i, s in enumerate(self.span_reference):
            if s is None:
                doc = Doc(self.vocab).from_disk(self.doc_path[i])
                self.span_reference[i] = doc[self.span_slice[i]]
        return self.span_reference

    def add_triplet(self, triplet: BeliefTriplet):
        self.sentence.append(triplet.sentence)
        self.confidence.append(triplet.confidence)
        self.span_slice.append(triplet.span_slice)
        self.doc_path.append(triplet.doc_path)
        self.span_reference.append(triplet.span_reference)

    def add_tg(self, triplet: TripletGroup):
        self.sentence += triplet.sentence
        self.confidence += triplet.confidence
        self.span_slice += triplet.span_slice
        self.doc_path += triplet.doc_path

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
