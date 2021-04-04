"""
test model loaders
"""

import pytest
from belief_graph.model_loaders import load_danish
from spacy.attrs import IS_SPACE

from .examples import EXAMPLES


@pytest.mark.parametrize("example", EXAMPLES)
def test_danish(example):
    nlp = load_danish()

    doc = nlp(example)

    for sent in doc.sents:
        print(sent)
