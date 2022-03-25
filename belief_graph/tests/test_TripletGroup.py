import os

import belief_graph as bg
from spacy.tokens import Span

from .test_BeliefTriplet import simple_triplets


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


def test_offload(simple_triplets):
    t1, t2 = simple_triplets
    tg = bg.TripletGroup.from_belief_triplet(t1)
    tg.add(t1)

    tg.offload()
    paths = tg.doc_path
    for path in paths:
        assert os.path.exists(path)
    assert all(s is None for s in tg.span_reference)
    assert all(isinstance(s, Span) for s in tg.span)
