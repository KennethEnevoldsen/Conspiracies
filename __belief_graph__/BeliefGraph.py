"""
The script containing the main belief graph class
"""

from typing import Union, List, Optional, Tuple

from BeliefTriplets import BeliefTriplets


class BeliefGraph():
    """
    a class
    """

    def __init__(self):
        self.belief_triplets = []

    def add_belief_triplets(self, triplet: BeliefTriplets):
        self.belief_triplets.append(BeliefTriplets)

    def filter_beliefs(
            threshold: Optional[float] = None,
            filter_non_continous: Optional[bool] = None,
            lemmatize_relations: Optional[bool] = None,
            lemmatize_head: Optional[bool] = None,
            lemmatize_tail: Optional[bool] = None
    ):
                if threshold is None:
            threshold == self.threshold
        if filter_non_continous is None:
            filter_non_continous == self.filter_non_continous
        if lemmatize_relations is None:
            lemmatize_relations == self.lemmatize_relations
        if lemmatize_head is None:
            lemmatize_head == self.lemmatize_head
        if lemmatize_tail is None:
            lemmatize_tail == self.lemmatize_tail

        for triplet in self.belief_triplets:
            triplet.filter_triplets(filter_non_continous=filter_non_continous,
                                    threshold=threshold,
                                    lemmatize_relations=lemmatize_relations,
                                    lemmatize_head=lemmatize_head,
                                    lemmatize_tail=lemmatize_tail)

    def plot_node():
        pass

    def extract_node_relations():
        pass
