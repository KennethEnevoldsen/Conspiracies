"""
"""
from typing import List, OrderedDict

import pytest
from belief_graph import BeliefTriplet, load_danish
from spacy.tokens import Span


@pytest.fixture
def simple_triplets() -> List[BeliefTriplet]:
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

    doc = nlp("Mit navn er Kenneth. .")
    span = next(doc.sents)

    path = (0, 1, 2)
    bt_ = BeliefTriplet(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )
    return [bt, bt_]


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
    assert isinstance(bt.head_span, Span)

    path = (3, 1, 2, 0)
    rev_bt = BeliefTriplet(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )

    assert bt < rev_bt, "belief triplets should be sorted according to their heads"

    bt.set_getter("lemma")
    print(bt)
