import belief_graph as bg

from .test_BeliefTriplet import simple_triplets
import os
from spacy.tokens import Span

def test_from_belief_triplet(simple_triplets):
    for triplet in simple_triplets:
        tg = bg.TripletGroup.from_belief_triplet(triplet)
        print(tg)


def test_add(simple_triplets):
    for t1, t2 in zip(simple_triplets, simple_triplets):
        tg = bg.TripletGroup.from_belief_triplet(t1)
        tg.add(t2)
    print(tg)

    tg_ = bg.TripletGroup.from_belief_triplet(t1)
    assert tg == tg_

    tg_.offload()
    path = tg_._doc_reference
    assert os.path.exists(path)
    assert all(s is None for s in tg_.__span)
    assert all(isinstance(s, Span) for s in tg_.span)
