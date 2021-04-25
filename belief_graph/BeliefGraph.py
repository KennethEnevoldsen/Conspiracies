"""
The script containing the main belief graph class
"""
from heapq import heappop, heappush
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Union
from warnings import warn

from .belief_extraction import BeliefParser
from .data_classes import BeliefTriplet, TripletGroup
from .filters import TripletFilter, TripletGroupFilter
from .BeliefNetwork import BeliefNetwork


class BeliefGraph:
    """"""

    def __init__(
        self,
        parser: BeliefParser,
        triplet_filters: List[TripletFilter] = [],
        group_filters: List[TripletGroupFilter] = [],
    ):
        """"""
        if isinstance(triplet_filters, TripletFilter):
            triplet_filters = [triplet_filters]
        if isinstance(group_filters, TripletFilter):
            group_filters = [group_filters]

        self.parser = parser
        self.triplet_filters = triplet_filters
        self.group_filters = group_filters
        self._triplet_heap = []
        self._triplet_group_heap = []

        self.is_sorted = True


    def add_belief_triplets(self, triplets: Iterable[BeliefTriplet]):
        self.is_sorted, self.is_filtered, self.is_merged = False, False, False
        for triplet in triplets:
            heappush(self._triplet_heap, triplet)

    def add_texts(self, texts: Union[Iterable[str], str]):
        triplets = self.parser.parse_texts(texts)
        self.add_belief_triplets(triplets)

    def _sort_triplets(self):
        """
        sorts the triplet heap
        """
        if self.is_sorted is True:
            return None
        ordered = []

        # While we have elements left in the heap
        while self._triplet_heap:
            ordered.append(heappop(self._triplet_heap))

        self._triplet_heap = ordered
        self.is_sorted = True


    def replace_filters(
        self,
        triplet_filters: Optional[List[TripletFilter]] = None,
        group_filters: Optional[List[TripletFilter]] = None,
    ):
        if triplet_filters is not None:
            self.triplet_filters = triplet_filters
        if group_filters is not None:
            self.group_filters = group_filters

    def plot_node(self, 
                  type: Union[List, str]="head", # not currently implemented
                  nodes: Union[List, str]="all",
                  scale_confidence: bool=False,
                  k=0.5, # tweak for spacing
                  save_name="none",
                  return_graph=False,
                  **kwargs):
        
        bn = BeliefNetwork(self)
        bn.construct_graph(nodes=nodes, scale_confidence=scale_confidence)
        bn.plot_graph(save_name=save_name, k=k, **kwargs)
        if return_graph:
            return bn


    def extract_node_relations():
        pass

    @staticmethod
    def merge_filters(filters: List[TripletFilter]) -> Callable:
        def wrapper(
            triplets: Iterable[BeliefTriplet],
        ) -> Generator[BeliefTriplet, None, None]:
            for f in filters:
                triplets = f.filter(triplets)
            return triplets

        return wrapper

    @property
    def triplet_groups(self) -> Generator[TripletGroup, None, None]:
        return merge_triplets(self.filtered_triplets)

    @property
    def triplets(self) -> List[BeliefTriplet]:
        if self.is_sorted is False:
            self._sort_triplets()
        return self._triplet_heap

    @property
    def filtered_triplets(self) -> Generator[BeliefTriplet, None, None]:
        triplet_filter = self.merge_filters(self.triplet_filters)
        return triplet_filter(self.triplets)

    @property
    def filtered_triplet_groups(self) -> Generator[TripletGroup, None, None]:
        triplet_filter = self.merge_filters(self.triplet_groups)
        return self.triplet_filter(self.triplet_groups)

def merge_triplets(
    sorted_triplets: List[BeliefTriplet],
) -> Generator[TripletGroup, None, None]:
    """
    sorted_triplets (List[BeliefTriplet]): Assumed a list of sorted triplets.
    """
    queue = sorted_triplets

    while queue:
        triplet = queue.pop()

        tg = TripletGroup.from_belief_triplet(triplet)
        try:
            while triplet == queue[-1]:
                tg.add_triplet(queue.pop())
        except IndexError:
            pass            
        yield tg
