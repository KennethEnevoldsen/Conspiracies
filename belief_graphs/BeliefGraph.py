"""
The script containing the main belief graph class
"""

from typing import Union, List, Optional, Tuple

from BeliefTriplet import BeliefTriplet


class BeliefGraph():
    """
    """

    def __init__(self):
        self.belief_triplets = []

    def add_belief_triplets(self, triplet: BeliefTriplet):
        self.belief_triplets.append(BeliefTriplet)

    def filter_beliefs(
            threshold: Optional[float] = None,
            filter_non_continous: Optional[bool] = None,
            lemmatize_relations: Optional[bool] = None,
            lemmatize_head: Optional[bool] = None,
            lemmatize_tail: Optional[bool] = None
    ):
        pass

    def plot_node():
        pass

    def extract_node_relations():
        pass