from belief_graph.data_classes.BeliefTriplet import BeliefTriplet
from typing import Callable, Generator, Iterable, List, Optional, Union


def make_simple_triplet_filter(
    valid_heads: List[Union[str, Callable]],
    valid_tails: List[Union[str, Callable]],
    valid_relations: List[Union[str, Callable]],
    reject_entire: bool = False,
    continous: bool = False
    
) -> Callable:
    """
    """
    def wrapper(triplet: BeliefTriplet) -> Union[BeliefTriplet, None]:
        pass
    return wrapper


def TripletFilter():
    def __init__(
        self,
        valid_heads: List[Union[str, Callable]],
        valid_tails: List[Union[str, Callable]],
        valid_relations: List[Union[str, Callable]],
        group_filters: List[Union[str, Callable]],
        list_filters: Optional[List[Callable]] = None,
        reject_entire: bool = False,
    ):
        """
        triplet_filters (): functions takes a list of triplet and only yield valid triplets. str is predefined filters. Which can be found
        in belief_graph.filters.PREDEFINED
        group_filters (): same as triplet_filters but taking a TripletGroup as argument
        list_filters (): A filter which takes in a list of TripletGroup. Note that the list is supplied as sorted in BeliefGraph.
        reject_entire (bool): Should the entire triplet be rejected if one token does not pass the filter or should the remainder of the relation tokens constitute a relation.
        continuous (bool): Should the relation be continuous. if
        """
        self.triplet_filters = triplet_filters
        self.group_filters = group_filters
        self.list_filters = list_filters

    @staticmethod
    def _filter_tokens(
        self, triplets: Iterable[BeliefTriplet], attr: str, funcs: List[Callable]
    ) -> Generator[BeliefTriplet]:
        for triplet in triplets:
            valid = True
            for f in funcs:
                token = getattr(triplet, attr)
                if not f(token):
                    valid = False
            if not valid:
                continue
            yield triplet

    def filter_heads(
        self, triplets: Iterable[BeliefTriplet]
    ) -> Generator[BeliefTriplet]:
        return self._filter_tokens(triplets, attr="head_token", funcs=self.valid_heads)

    def filter_tails(
        self, triplets: Iterable[BeliefTriplet]
    ) -> Generator[BeliefTriplet]:
        return self._filter_tokens(triplets, attr="tail_token", funcs=self.valid_tails)

    def filter_relations(
        self,
        triplets: Iterable[BeliefTriplet],
    ) -> Generator[BeliefTriplet]:
        for triplet in triplets:
            reject_triplet = False
            keep = []
            for i, t in enumerate(triplet.relation_list):
                passed = all(f(t) for f in funs)
                if not passed:
                    if reject_entire:
                        reject_triplet = True
                        break
                    keep.append(False)
                else:
                    keep.append(True)

            if reject_triplet:
                continue
            elif passed:
                yield triplet
            else:
                if continuous is True and not is_cont(triplet._relation_ids):
                    continue

                triplet._relation_ids = tuple(
                    i for i, k in zip(triplet._relation_ids, keep) if k
                )
                yield triplet