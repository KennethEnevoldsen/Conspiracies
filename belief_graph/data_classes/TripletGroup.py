from __future__ import annotations

from typing import List, Optional, Union

from spacy.tokens import Span
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
    __vocab: Optional[Vocab] = None
    _span_slice: List[slice] = []
    _doc_reference: List[str] = []
    __span: List[Union[None, Span]] = []


    @property
    def count(self):
        return len(self.sentence)

    @staticmethod
    def from_belief_triplet(triplet: BaseTriplet, offload=False) -> TripletGroup:
        tg = TripletGroup(
            head = triplet.head,
            tail = triplet.tail,
            sentence = [triplet.sentence],
            confidence = [triplet.confidence]
        )
        if not isinstance(triplet, BeliefTriplet):
            return tg
        if offload is True:
            triplet.offload()

        tg._span_slice.append(triplet._span_slice)
        tg._doc_reference.append(triplet._doc_reference)

        if triplet.is_offloaded is True:  # if it is offloaded
            tg.__vocab = triplet.__vocab
            tg.__span.append(None)
        else:
            tg.__span.append(triplet.span)

    @property
    def is_offloaded(self) -> bool:
        return all(s is None for s in self.__span)

    def offload(self, vocab: Optional[Vocab] = None, id_set: Optional[set] = None, dir="triplet_docs"):
        if vocab is not None:
            self.vocab = vocab
        if self.vocab is None:
            raise ValueError("No vocab is specified. Writing a doc without a Vocab will lead to the doc not being properly recoverable")
        
        for i, s in enumerate(self.__span):
            if s is None:
                continue
            path = spacy_doc_to_dir(s.doc, id_set = id_set, dir=dir)
            self._span_slice[i] = slice(s.start, s.end)
            self._doc_reference[i] = path

    def add_triplet(self, triplet: BeliefTriplet):
        self.sentence.apend(triplet.sentence)
        self.confidence.append(triplet.confidence)
        self._span_slice.append(triplet._span_slice)
        self._doc_reference.append(triplet._doc_reference)

    def add_tg(self, triplet: TripletGroup):
        self.sentence += triplet.sentence
        self.confidence += triplet.confidence
        self._span_slice += triplet._span_slice
        self._doc_reference += triplet._doc_reference

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
