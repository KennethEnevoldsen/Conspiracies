"""
a series of extension to the spacy doc
"""
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.attrs import IS_SPACE


def doc_wp2tokid_getter(doc: Doc, bos=True, eos=True):
    """
    extract the wordpiece2tokenID mapping from a doc
    create a mapping from wordpieces to tokens id in the form of a list
    e.g.
    [0, 1, 1, 1, 2]
    indicate that there are three tokens (0, 1, 2) and token 1 consist of three
    wordpieces
    note: this only works under the assumption that the word pieces
    are trained using similar tokens. (e.g. split by whitespace)

    example:
    Doc.set_extension("wp2tokid", getter=doc_wp2tokid_getter)
    """
    wp2tokid = []
    tok = 0
    if bos is True:
        wp2tokid.append(None)
    for i in doc._.trf_data.align.lengths:
        wp2tokid += [tok] * i
        tok += 1
    if eos is True:
        wp2tokid.append(None)
    return wp2tokid


def doc_tokid2wp_getter(doc: Doc):
    """
    extract the tokenID2wordpiece mapping from a doc

    example:
    Doc.set_extension("tokid2wp", getter=doc_tokid2wp_getter)
    """
    return {
        tok_idx: wp_idx for wp_idx, tok_idx in enumerate(doc._.wp2tokid) if not None
    }


def span_wp2tokid_getter(span: Span):
    """
    extract span specific wp2tokid mapping
    """
    tokid2wp = span.doc._.tokid2wp
    s = tokid2wp[span.start]
    _ = span.end
    while span.doc[_].check_flag(IS_SPACE):
        _ -= 1
    e = tokid2wp[_]
    return span.doc._.wp2tokid[s:e]


def span_attn_getter(span: Span, layer=-1):
    """
    extract the attention matrix for the tokens in the sentence
    """
    attn = span.doc._.trf_data.attention[layer]

    tokid2wp = span.doc._.tokid2wp
    s = tokid2wp[span.start]
    _ = span.end
    while span.doc[_].check_flag(IS_SPACE):
        _ -= 1
    e = tokid2wp[_]

    return attn[:, :, s:e, s:e]