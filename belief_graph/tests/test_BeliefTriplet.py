"""
"""
from spacy.tokens import Span
from belief_graph import BeliefTriplet, load_danish


def test_BeliefTriplet():
    nlp = load_danish()
    doc = nlp("Dette består af to sætninger")
    span = next(doc.sents)

    bt = BeliefTriplet(path=(0, 1, 2, 3), span=span, confidence=1)

    assert len(bt.relation_list) == 2
    assert isinstance(bt.confidence, float)
    assert isinstance(bt.head_token, Span)