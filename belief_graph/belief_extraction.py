"""
This script contains function for extracting/parsing beliefs (a
proposed knowledge triplet) from a text
"""
from typing import List, Iterable, Union
from functools import partial

from numpy import ndarray
import numpy as np

from transformers import PreTrainedTokenizerBase
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.language import Language

from pydantic import validate_arguments

from .utils import merge_token_attention, attn_to_graph, beam_search
from .BeliefTriplet import BeliefTriplet


def extract_attention(doc, layer=-1):
    return doc._.trf_data.attention[layer]


def extract_wordpieces(doc: Doc, remove_bos: bool = True, remove_eos: bool = True):
    """"""
    bos = 1 if remove_bos else 0
    eos = -1 if remove_bos else None
    return doc._.trf_data.wordpieces.input_ids[bos:eos]


def aggregate_noun_chunks(sent_span: Span) -> List:
    """
    return (List) a list spacy tokens and spacy spans where a span corresponds to a noun_chunk
    """
    start = sent_span.start
    i = 0
    out = []
    for nc in sent_span.noun_chunks:
        s = nc.start - start
        if s != 0:
            out + [t for t in sent_span[i:s]]
        out.append(nc)
        i = nc.end - start
    return out


class BeliefParser:
    """
    A class for extracting belief triplets from a spacy doc
    """

    def __init__(
        self,
        nlp: Language,
        n_beams: int = 6,
        max_length=None,
        min_length: int = 3,
        alpha: float = 1,
        num_return_paths: int = 1,
        filter_non_continous: bool = True,
        aggregate_method: str = "mult",
        attn_layer: int = -1,
    ):
        self.nlp = nlp
        self.n_beams = n_beams
        self.max_length = max_length
        self.min_length = min_length
        self.alpha = alpha
        self.num_return_paths = num_return_paths
        self.filter_non_continous = filter_non_continous
        self.aggregate_method = aggregate_method
        self.attn_layer = attn_layer

        self.beam_search = partial(
            beam_search,
            n_beams=n_beams,
            alpha=alpha,
            max_length=max_length,
            min_length=min_length,
            num_return_paths=num_return_paths,
            aggregate_method=aggregate_method,
        )

    def parse_texts(self, texts: Union[Iterable[str], str]):
        """
        text (Union[Iterable[str], str]): An iterable object (e.g. a list) of string or a simply list a string
        """
        if isinstance(texts, str):
            texts = [texts]

        docs = self.nlp.pipe(texts)
        for doc in docs:
            for parse in self.parse_doc(doc):
                yield parse

    def parse_doc(self, doc: Doc):
        """
        doc (Doc): A SpaCy Doc
        """
        for sent in doc.sents:
            for parse in self.parse_sentence(sent):
                yield parse

    def parse_sentence(self, sent_span: Span):
        """
        sentence_span (Span): a SpaCy sentence span
        """

        wordpiece2nctoken_id = sent_span._.wp2ncid
        attn = sent_span._.attention[0]

        agg_attn = np.mean(attn, axis=0)

        merged_attn = merge_token_attention(agg_attn, wordpiece2nctoken_id)

        # make a forward and backward attention graph
        backward_attn_graph, forward_attn_graph = attn_to_graph(merged_attn)

        # create head tail pair
        tail_head_pairs = []
        for h, nc in enumerate(sent_span.noun_chunks):
            for t, nc in enumerate(sent_span.noun_chunks):
                if h != t:
                    tail_head_pairs.append((h, t))

        def beam_search_(args):
            head, tail = args
            graph = forward_attn_graph if head < tail else backward_attn_graph
            return self.beam_search(head, tail, graph=graph)

        relation_pairs = []
        for output in map(beam_search_, tail_head_pairs):
            if len(output):
                for path, conf in output:
                    yield BeliefTriplet(path=path, confidence=conf, span=sent_span)
