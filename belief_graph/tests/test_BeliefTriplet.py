"""
"""

from belief_graph import BeliefTriplet, load_danish
from spacy.tokens import Span


def simple_triplet():
    nlp = load_danish()
    doc = nlp("Dette består af to sætninger")
    span = next(doc.sents)

    path = (0, 1, 2, 3)
    bt = BeliefTriplet(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )
    return bt


def test_BeliefTriplet():
    nlp = load_danish()
    doc = nlp("Dette består af to sætninger")
    span = next(doc.sents)

    path = (0, 1, 2, 3)
    bt = BeliefTriplet(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )

    assert len(bt.relation_list) == 2
    assert isinstance(bt.confidence, float)
    assert isinstance(bt.head_token, Span)

    path = (3, 1, 2, 0)
    rev_bt = BeliefTriplet(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )

    assert bt < rev_bt, "belief triplets should be sorted according to their heads"
