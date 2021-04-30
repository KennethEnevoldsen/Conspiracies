# import sys
# sys.path.append("/Users/au561649/Desktop/Github/UCLA-Conspiracies")

import pytest
from belief_graph.model_loaders import load_danish
from spacy.attrs import IS_SPACE
from spacy.language import Language

from .examples import EXAMPLES


@pytest.fixture
def nlp_model() -> Language:
    nlp = load_danish()
    return nlp


@pytest.mark.parametrize("example", EXAMPLES)
def test_danish(example, nlp_model):
    nlp = nlp_model

    doc = nlp(example)

    for sent in doc.sents:
        print(sent)


def test_wp_accents(nlp_model):
    """
    tests if the transformers deals with special language characters
    """
    nlp = nlp_model
    doc = nlp("æøå")
    wp = doc._.trf_data.wordpieces.strings[0]
    for letter in "æøå":
        assert [i for i in wp if letter in i]


@pytest.mark.parametrize("example", EXAMPLES)
def test_doc_extention_wp2tokid(example, nlp_model):
    nlp = nlp_model
    doc = nlp(example)

    # wp2tokid should be the same
    assert len(doc._.wp2tokid) == doc._.trf_data.wordpieces.lengths[0]

    # length of document tokens shoudl be the same of the number of unique tokens
    non_space = [t for t in doc if not t.check_flag(IS_SPACE)]
    assert len(non_space) == len(set(filter(lambda x: x is not None, doc._.wp2tokid)))


@pytest.mark.parametrize("example", EXAMPLES)
def test_doc_extention_tokid2wp(example, nlp_model):
    nlp = nlp_model
    doc = nlp(example)
    # doc = nlp("Dette består af to sætninger. Kan dette være problematisk? Genåbning")
    assert len(doc) == len(doc._.tokid2wp)

    wp = doc._.trf_data.wordpieces.strings[0]

    tokid2wp = doc._.tokid2wp

    for i in range(10):
        print(doc[i])
        if tokid2wp[i] is None:
            continue
        print("\t WP:", wp[tokid2wp[i]])


@pytest.mark.parametrize("example", EXAMPLES)
def test_doc_extention_tokid2wp(example, nlp_model):
    nlp = nlp_model
    doc = nlp(example)
    # doc = nlp("\nDette består af to sætninger. Kan dette være problematisk? Genåbning")
    assert len(doc) == len(doc._.tokid2wp)
    assert len(doc) == len(doc._.tokid2nc)
    assert len(doc) == len(doc._.tokid2ncid)

    assert len(doc._.nctokens) <= len(doc)

    wp = doc._.trf_data.wordpieces.strings[0]
    assert len(wp) <= len(doc._.wp2ncid)
    assert len(wp) <= len(doc._.wp2tokid)

    # wp match up
    tokid2wp = doc._.tokid2wp
    for i in range(3):
        print(doc[i])
        if tokid2wp[i] is None:
            continue
        print("\t WP:", wp[tokid2wp[i]])
        print("\t WP:", wp[tokid2wp[i]])


@pytest.mark.parametrize("example", EXAMPLES)
def test_sent_extention(example, nlp_model):
    nlp = nlp_model
    doc = nlp(example)

    for span in doc.sents:
        print(span)
        assert len(span._.wordpieces) == len(span._.wp2tokid)
        assert len(span._.wordpieces) == span._.attention.shape[-1]
        assert len(span._.wordpieces) == len(span._.wp2ncid)
        assert len(span._.nctokens) == len(set(span._.tokid2ncid))

