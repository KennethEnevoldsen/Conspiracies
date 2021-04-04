"""
The script containing the main belief graph class
"""
from heapq import heappop, heappush
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Union
from warnings import warn

from .belief_extraction import BeliefParser
from .data_classes import BeliefTriplet, TripletGroup
from .filters import TripletFilter


class BeliefGraph:
    """"""

    def __init__(self, parser: BeliefParser, filter: TripletFilter):
        """
        filter ():
        """
        self.parser = parser
        self.filter = filter
        self._triplet_heap = []
        self._triplet_group_heap = []

        self.is_sorted = True
        self.is_filtered = True
        self.is_group_filtered = True
        self.is_merged = True

        self.__filtered_triplets = []
        self.__triplet_groups = []
        self.__filtered_triplet_groups = []

    def add_belief_triplets(self, triplets: Iterable[BeliefTriplet]):
        self.is_sorted, self.is_filtered, self.is_merged = False, False, False
        for triplet in triplets:
            heappush(self._triplet_heap, triplet)

    def add_texts(self, texts: Union[Iterable[str], str]):
        triplets = self.parser.parse_texts(texts)
        self.add_belief_triplet(triplets)

    def _sort_triplets(self):
        """
        sorts the triplet heap
        """
        if self._sorted is True:
            return None
        ordered = []

        # While we have elements left in the heap
        while self._triplet_heap:
            ordered.append(heappop(self._triplet_heap))

        self._triplet_heap = ordered
        self.is_sorted = True

    def __merge_triplets(self):
        if self.is_merged:
            return None
        else:
            self.__triplet_groups = merge_triplets(self.filtered_triplets)
        self.is_merged = True

    def replace_filter(self, filter: TripletFilter):
        self.filter = filter

    def plot_node():
        pass

    def extract_node_relations():
        pass

    @property
    def triplet_groups(self):
        self.__merge_triplets()
        return self.__triplet_groups

    @property
    def triplets(self):
        if self.sorted is False:
            self._sort_triplets()
        return self.triplets

    @property
    def filtered_triplets(self):
        if self.is_filtered is True:
            self.__filtered_triplets
        else:
            self.__filtered_triplets = self.filter.filter_triplets(self.triplets)
            self.is_filtered = True

    @property
    def filtered_triplet_groups(self):
        if self.is_group_filtered is True:
            self.__filtered_triplet_groups
        else:
            self.__filtered_triplet_groups = self.filter.filter_groups(
                self.triplet_groups
            )
            self.is_group_filtered = True


def merge_triplets(sorted_triplets: List[BeliefTriplet]) -> Generator:
    """
    sorted_triplets (List[BeliefTriplet]): Assumed a list of sorted triplets.
    """
    queue = sorted_triplets

    while queue:
        triplet = queue.pop()

        tg = TripletGroup.from_belief_triplet(triplet)
        while triplet == queue[-1]:
            tg.__add_triplet(queue.pop())
        yield tg
