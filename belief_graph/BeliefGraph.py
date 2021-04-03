"""
The script containing the main belief graph class
"""
from heapq import heappop, heappush
from warnings import warn

from typing import Callable, Generator, Union, List, Optional, Tuple

from .data_classes import BeliefTriplet, TripletGroup
from .filters import TripletFilter

class BeliefGraph():
    """
    """

    def __init__(self, triplet_filters: List, tripletgroup_filters: List):
        """
        filters (): A list of filters to apply. Should be callable or a tring in the lsit predefined filters. Which can be found
        in belief_graph.filters.PREDEFINED.
        """
        self._triplet_heap = []
        self._sorted = True
        self._filtered_triplets = None

    def add_belief_triplets(self, triplet: BeliefTriplet):
        self._sorted = False
        heappush(self._triplet_heap, BeliefTriplet)

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
        self._sorted = True      


    def add_new_filter(tf: TripletFilter, funs = Optional[List[Callable]]=None):
        """
        tf (): A triplet filter
        funs (): Optional function to filter the list of triplets
        """
        pass

    def plot_node():
        pass

    def extract_node_relations():
        pass

    @property
    def grouped_triplets(self):
        return merge_triplets(self.triplets)

    @property
    def triplets(self):
        if self.sorted is False:
            self._sort_triplets()
        return self.triplets
    
    @property
    def filtered_triplets(self):
        pass



def merge_triplets(sorted_triplets: List[BeliefTriplet]) -> Generator:
    """
    sorted_triplets (List[BeliefTriplet]): Assumed a list of sorted triplets.
    """
    queue = sorted_triplets

    while queue:
        triplet = queue.pop()

        tg = TripletGroup.from_belief_triplet(triplet)
        while triplet == queue[-1]:
            tg.add(queue.pop())
        yield tg
            

