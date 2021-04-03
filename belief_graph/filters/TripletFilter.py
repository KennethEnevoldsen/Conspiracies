from ..data_classes import BeliefTriplet, TripletGroup
from typing import Callable, Generator, Iterable, List, Optional, Union


def TripletFilter():
    def __init__(
        self,
        triplet_filters = List[Callable],
        group_filters = List[Callable]
        ):
        """
        triplet_filters (): functions takes a BeliefTriplet and return the BeliefTriplet (or modifed version thereof) or None if the triplet should be discarded.
        group_filters (): same as triplet_filters but taking a TripletGroup as argument
        list_filters (): A filter which takes in a list of TripletGroup. Note that the list is supplied as sorted in BeliefGraph.
        reject_entire (bool): Should the entire triplet be rejected if one token does not pass the filter or should the remainder of the relation tokens constitute a relation.
        continuous (bool): Should the relation be continuous. if
        """
        self.triplet_filters = triplet_filters
        self.group_filters = group_filters

    def filter_triplets(self, triplets: Iterable[BeliefTriplet]) -> Generator[BeliefTriplet]:
        for triplet in triplets:
            is_none = False
            for func in self.triplet_filters:
                triplet = func(triplet)
                
                if triplet is None:
                    is_none = True
                    break
            
            if is_none is True:
                continue
            yield triplet


    def filter_group(self, triplets: Iterable[TripletGroup]) -> Generator[TripletGroup]:
        for triplet in triplets:
            is_none = False
            for func in self.group_filters:
                triplet = func(triplet)
                
                if triplet is None:
                    is_none = True
                    break
            
            if is_none is True:
                continue
            yield triplet
