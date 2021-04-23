from typing import List, OrderedDict


import transformers
from belief_graph import BeliefTriplet, load_danish
from spacy.tokens import Span
import os


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

bt.span_reference

dir(bt)
bt

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

bt.offload()
path = bt._doc_reference
assert os.path.exists(path)

assert bt.__span is None
assert isinstance(bt.span, Span)
