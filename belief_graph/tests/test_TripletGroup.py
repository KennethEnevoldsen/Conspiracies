
import belief_graph as bg

from .test_BeliefTriplet import simple_triplets


def test_from_belief_triplet(simple_triplets):
    for triplet in simple_triplets:
        bg.TripletGroup.from_belief_triplet(triplet)

def test_add(simple_triplets):
    triplets = simple_triplets + simple_triplets
    for triplet in triplets:
        bg.TripletGroup.from_belief_triplet(triplet)