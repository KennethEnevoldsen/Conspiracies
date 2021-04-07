from functools import partial
from typing import Callable, Generator, Iterable, List, Optional, Union

from pydantic import BaseModel, validate_arguments
from spacy.tokens import Span

from ..data_classes import BeliefTriplet, TripletGroup
from .triplet_filters import (
    filter_confidence,
    filter_head,
    filter_is_continuous,
    filter_relations,
    filter_tail,
    valid_dependency,
    valid_entities,
    valid_pos,
)
from .tripletgroup_filters import count_filter


class TripletFilter(BaseModel):
    """
    filter for triplets
    """

    filter_func: Optional[Callable] = None

    @staticmethod
    def from_func(func: Callable):
        """
        func (): functions takes a BeliefTriplet and return the BeliefTriplet (or modifed version thereof) or None if the triplet should be discarded.
        """
        return TripletFilter(filter_func=func)

    def is_valid(self, triplet: BeliefTriplet):
        if isinstance(triplet, BeliefTriplet):
            triplet = self.func(triplet)
            if triplet is None:
                return False
            return True
        raise ValueError(
            f"Filter takes {BeliefTriplet}, but you applied it to {type(triplet)}"
        )

    @property
    def func(self) -> Callable:
        if self.filter_func is None:
            try:
                self.make_filter_func()
            except AttributeError as e:
                raise AttributeError(
                    f"{e}. This is likely caused as filter_func is left as None. We suggest using the TripletFilter.from_func method"
                )
        return self.filter_func

    def filter(
        self, triplets: Iterable[BeliefTriplet]
    ) -> Generator[Union[BeliefTriplet, None], None, None]:
        if isinstance(triplets, BeliefTriplet):
            triplets = [triplets]

        for triplet in triplets:
            is_none = False
            triplet = self.func(triplet)

            if triplet is None:
                is_none = True
                continue
            yield triplet


class SetFilter(TripletFilter):
    """
    this is a utility class for the PosFilter, DepFilter and EntFilter.
    It is not mean to be used on its own
    """

    valid: set = None
    invalid: set = None
    reject_entire: bool = False
    apply_to: set = {"head", "tail", "relation"}

    def make_func(self, func):
        funcs = []
        if "head" in self.apply_to:
            fh = partial(filter_head, func=func)
            funcs.append(fh)
        if "tail" in self.apply_to:
            ft = partial(filter_tail, func=func)
            funcs.append(ft)
        if "relation" in self.apply_to:
            fr = partial(filter_relations, func=func)
            funcs.append(fr)

        self.filter_func = self._merge_funcs(funcs)

    @staticmethod
    def _merge_funcs(funcs: List[Callable]) -> Callable:
        def wrapper(triplet: BeliefTriplet) -> Union[BeliefTriplet, None]:
            for func in funcs:
                triplet = func(triplet)
                if triplet is None:
                    return None
            return triplet

        return wrapper


class PosFilter(SetFilter):
    valid: set = None
    invalid: set = {"PUNCT", "SPACE", "NUM"}
    apply_to: set = {"head", "tail", "relation"}

    def make_filter_func(self):
        is_valid = partial(valid_pos, valid=self.valid, invalid=self.invalid)
        self.make_func(is_valid)


class EntFilter(SetFilter):
    valid: set = {"LOC", "PER", "ORG"}
    invalid: set = None
    apply_to: set = {"head", "tail"}

    def make_filter_func(self):
        is_valid = partial(valid_entities, valid=self.valid, invalid=self.invalid)
        self.make_func(is_valid)


class DepFilter(SetFilter):
    valid: set = None
    invalid: set = None
    apply_to: set = {"head", "tail"}

    def make_filter_func(self):
        is_valid = partial(valid_dependency, valid=self.valid, invalid=self.invalid)
        self.make_func(is_valid)


class ContinuousFilter(TripletFilter):
    """
    filter triplets which relations it not continuous
    """

    filter_func: Optional[Callable] = filter_is_continuous


class ConfidenceFilter(TripletFilter):
    threshold: float

    def make_filter_func(self):
        self.filter_func = partial(filter_confidence, threshold=self.threshold)


def lemma_getter(span: Span):
    return " ".join(t.lemma_ for t in span)


class LemmatizationFilter(TripletFilter):
    @staticmethod
    def filter_func(triplet: BeliefTriplet) -> BeliefTriplet:
        triplet.getter = lemma_getter


### --- TripletGroupFilters --- ###


class TripletGroupFilter(TripletFilter):
    """
    filter for triplets
    """

    def is_valid(self, triplet: TripletGroup):
        if isinstance(triplet, TripletGroup):
            triplet = self.func(triplet)
            if triplet is None:
                return False
            return True
        raise ValueError(
            f"Filter takes {TripletGroup}, but you applied it to {type(triplet)}"
        )

    def filter(
        self, triplets: Iterable[TripletGroup]
    ) -> Generator[TripletGroup, None, None]:
        return super().filter(triplets)


class CountFilter(TripletGroupFilter):
    count: float

    def make_filter_func(self):
        self.filter_func = partial(count_filter, count=self.count)
