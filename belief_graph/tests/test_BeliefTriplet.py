"""
"""
from typing import List, OrderedDict

import pytest
import transformers
from belief_graph import BeliefTriplet, load_danish
from spacy.tokens import Span
import os

@pytest.fixture
def simple_triplets() -> List[BeliefTriplet]:
    nlp = load_danish(transformer=None)
    doc = nlp("Dette består af to sætninger")
    span = next(doc.sents)

    path = (0, 1, 2, 3)
    bt = BeliefTriplet.from_parse(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )

    doc = nlp("Mit navn er Kenneth. .")
    span = next(doc.sents)

    path = (0, 1, 2)
    bt_ = BeliefTriplet.from_parse(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )
    return [bt, bt_]


def test_BeliefTriplet():
    nlp = load_danish(transformer=None)
    doc = nlp("Dette består af to sætninger")
    span = next(doc.sents)

    path = (0, 1, 2, 3)
    bt = BeliefTriplet.from_parse(
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
    rev_bt = BeliefTriplet.from_parse(
        head_id=path[0],
        relation_ids=path[1:-1],
        tail_id=path[-1],
        span=span,
        confidence=1,
    )

    assert bt < rev_bt, "belief triplets should be sorted according to their heads"

def test_offload(simple_triplets):
    for triplet in simple_triplets:
        triplet.offload()
        path = triplet.doc_path
        assert os.path.exists(path)

        assert triplet.span_reference is None
        assert isinstance(triplet.span, Span)
