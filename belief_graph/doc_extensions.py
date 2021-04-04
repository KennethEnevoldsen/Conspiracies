"""
a series of extension to the spacy doc
"""
from typing import Iterable, List

from spacy.attrs import IS_SPACE
from spacy.tokens import Doc
from spacy.tokens.span import Span


def doc_wp2tokid_getter(doc: Doc, bos=True, eos=True) -> List:
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


def find_prev_val(idx: int, l: List) -> int:
    """
    finds previous values which is not None. If there is no previous non-None value it returns l[idx]
    """
    i = idx
    prev = None
    while prev is None:
        i -= 1
        if i < 0:
            return l[idx]
        prev = l[i]
    return prev


def doc_wp2ncid_getter(doc: Doc) -> List:
    """
    extract the "wordpiece to noun chunk ID"-mapping from a doc
    """
    wp2ncid = doc._.wp2tokid.copy()
    for nc in doc.noun_chunks:
        wp_slice = nc._.wp_slice
        wp2ncid_slice = wp2ncid[wp_slice]

        wp2ncid[wp_slice] = [wp2ncid_slice[0]] * len(wp2ncid_slice)
    return wp2ncid


def doc_nctokens_getter(doc: Doc) -> List:
    """
    extract the noun chunk token spans from a doc.
    this is the token list with the noun chunks collapsed to one "token"
    """
    slices = set()
    nctokens = []
    for s in doc._.tokid2nc:
        sl = (s.start, s.end)
        if sl in slices:
            continue
        slices.add(sl)
        nctokens.append(doc[sl[0] : sl[1]])
    return nctokens


def doc_tokid2nc_getter(doc: Doc) -> List:
    """
    extract the "token ID  to noun chunk tokens spans"-mapping from a doc.
    """
    tokid2nc = [doc[i : i + 1] for i in range(len(doc))]
    for nc in doc.noun_chunks:
        if len(nc) > 1:
            nc_slice = slice(nc.start, nc.end)
            tokid2nc[nc_slice] = [doc[nc_slice]] * len(tokid2nc[nc_slice])
    return tokid2nc


def doc_tokid2ncid_getter(doc: Doc) -> List:
    """
    extract the "token ID  to noun chunk tokens spans"-mapping from a doc.
    """
    tokid2nc = doc._.tokid2nc
    tokid2ncid = []
    for i, span in enumerate(tokid2nc):
        if i != 0 and tokid2nc[i - 1] == span:
            tokid2ncid.append(i - 1)
        else:
            tokid2ncid.append(i)
    return tokid2ncid


def doc_wp2ncid_getter(doc: Doc) -> List:
    """
    extract the "wordpiece ID to noun chunk tokens id"-mapping from a doc.
    """
    wp2tokid = doc._.wp2tokid
    tokid2ncid = doc._.tokid2ncid

    wp2ncid = wp2tokid.copy()
    for i, tokid in enumerate(wp2ncid):
        if tokid is None:
            continue
        wp2ncid[i] = tokid2ncid[tokid]
    return wp2ncid


def doc_tokid2wp_getter(doc: Doc, bos=True) -> List[slice]:
    """
    extract the tokenID2wordpiece mapping from a doc

    example:
    Doc.set_extension("tokid2wp", getter=doc_tokid2wp_getter)
    """
    tokid2wp = []
    n = 0
    if bos is True:
        n += 1
    for t in doc._.trf_data.align.lengths:
        tokid2wp.append(slice(n, n + t))
        n += t
    return tokid2wp


def span_wp_slice_getter(span: Span) -> slice:
    """
    extract span slide for wordpieces
    """
    tokid2wp = span.doc._.tokid2wp
    tokid2wp[span.start]
    span_slices = tokid2wp[span.start : span.end]
    s = span_slices[0].start
    e = span_slices[-1].stop
    return slice(s, e)


def span_wp2tokid_getter(span: Span) -> List:
    """
    extract span specific wp2tokid mapping
    """
    wp_slice = span._.wp_slice
    return span.doc._.wp2tokid[wp_slice]


def span_wp2ncid_getter(span: Span) -> List:
    """
    extract the "wordpiece to noun chunk ID"-mapping from a span
    """
    wp_slice = span._.wp_slice
    return span.doc._.wp2ncid[wp_slice]


def span_attn_getter(span: Span, layer=-1):
    """
    extract the attention matrix for the tokens in the sentence
    """
    attn = span.doc._.trf_data.attention[layer]
    wp_slice = span._.wp_slice

    return attn[:, :, wp_slice, wp_slice]


def span_wp_getter(span: Span) -> List[str]:
    """
    extract the wordpieces for the tokens in the sentence
    """
    wp = span.doc._.trf_data.wordpieces.strings[0]
    wp_slice = span._.wp_slice
    return wp[wp_slice]


def span_nctokens_getter(span: Span) -> List[str]:
    """
    extract the noun chunk tokens from the span
    """
    s, e = span.start, span.end
    tokid2ncid = span.doc._.tokid2ncid[s:e]
    s, e = tokid2ncid[0], tokid2ncid[-1]
    return span.doc._.nctokens[s : e + 1]
