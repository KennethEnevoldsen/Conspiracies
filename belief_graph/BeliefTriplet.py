"""
This Script contains the Belief Triplet class
"""
from typing import Tuple
from pydantic import BaseModel


class BeliefTriplet(BaseModel):
    """
    - contains
        - triplets
        - conf
        - (Nc converter (Noun Chunk to everything))

    Examples:
    >>> t = BeliefTriplet(triplet=("Kenneth", "lastname", "Enevoldsen"), confidence=0.5)
    >>> t.tail
    "Enevoldsen"
    """
    triplet: Tuple[str, str, str]
    confidence: float

    @property
    def head(self):
        return self.triplet[0]

    @property
    def relation(self):
        return self.triplet[1]

    @property
    def tail(self):
        return self.triplet[-1]
