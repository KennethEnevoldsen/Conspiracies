"""
test model loaders
"""
# import sys
# sys.path.append("/Users/au561649/Desktop/Github/UCLA-Conspiracies")


import pytest

from spacy.attrs import IS_SPACE

from belief_graph.model_loaders import load_danish
from .examples import EXAMPLES


@pytest.mark.parametrize("example", EXAMPLES)
def test_danish(example):
    nlp = load_danish()

    doc = nlp(example)

    for sent in doc.sents:
        print(sent)
