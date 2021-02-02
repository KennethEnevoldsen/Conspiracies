"""
"""
from typing import Union

from functools import partial

from utils import (
    create_wordpiece_token_mapping,
    attn_to_graph,
    merge_token_attention,
    aggregate_attentions_heads,
    trim_attention_matrix,
    beam_search
)
from utils import is_a_range as is_continous


def create_mapping(
        tokens,
        noun_chunks,
        noun_chunk_token_span,
        tokenizer):
    """
    tokenizer: a huggingface tokenizer
    Creates mappings from token id to its tokens as its tags.
    it also creates a mapping from a token to the tokenizer id

    wordpiece2token: wordpiece -> Noun chunk merged tokens


    Example:
    from utils import load_example
    mappings = create_mapping(**load_example())
    """

    start_chunk = {s: e for s, e in noun_chunk_token_span}

    nc_tokens = []  # noun chunk merged token list
    nc_tokens_id2tokens_id = {}
    noun_chunk_w_id = []  # noun chunk with id

    i = 0
    chunk_id = 0
    while i < len(tokens):
        id_ = len(nc_tokens)
        if i in start_chunk:
            nc_tokens.append(noun_chunks[chunk_id])
            nc_tokens_id2tokens_id[id_] = noun_chunk_token_span[chunk_id]
            noun_chunk_w_id.append((noun_chunks[chunk_id], id_))
            chunk_id += 1
            i = start_chunk[i]  # end chunk
        else:  # if not in chunk
            nc_tokens.append(tokens[i])
            nc_tokens_id2tokens_id[id_] = i
            i += 1

    wordpiece2token = create_wordpiece_token_mapping(
        tokens=nc_tokens, tokenizer=tokenizer)

    return wordpiece2token, nc_tokens_id2tokens_id, noun_chunk_w_id


class NounChunkTokenIDConverter():
    def __init__(self,
                 nc_tokens_id2tokens_id, tokens,
                 lemmas, pos, dependencies, ner
                 ):
        self.tags = {
            "token": tokens,
            "lemma": lemmas,
            "pos": pos,
            "dependencies": dependencies,
            "ner": ner}
        self.nc_tokens_id2tokens_id = nc_tokens_id2tokens_id
        self.invalid = {}

    def add_invalid(self, invalid: set,
                    tag: str = "pos"):
        self.invalid[tag] = invalid

    def convert_to_str(self, nc_tok_id, tag="token"):
        tok_id = self.nc_tokens_id2tokens_id[nc_tok_id]

        # if it is a noun chunk
        if isinstance(tok_id, (list, tuple)):
            return convert_nc_to_str(nc_tok_id, tag)

        return self.tags[tag][tok_id] if self.is_tok_id_valid(tok_id) else None

    def is_tok_id_valid(self, tok_id):
        if self.invalid:
            for k in self.invalid.keys():
                if self.tags[k][tok_id] in self.invalid[k]:
                    return False
        return True

    def convert_nc_to_str(self, nc_tok_id, tag="token"):
        tok_id = self.nc_tokens_id2tokens_id[nc_tok_id]

        if self.invalid:
            r = [self.tags[k][tok_id] for i in range(tok_id[0], tok_id[1])]
            if len(r) == 0:
                return None
            return " ".join(r)
        return " ".join(self.tags[tag][tok_id[0]: tok_id[1]])


def triplet_to_str(triplet: Union[list, tuple],
                   nc_converter,
                   lemmatize_relations: bool = False,
                   lemmatize_head: bool = False,
                   lemmatize_tail: bool = False,
                   invalid_pos: set = set(),
                   invalid_dep: set = set(),
                   ):
    """
    """
    head, relation, tail = triplet[0], triplet[1:-1], triplet[-1]

    tag = "lemma" if lemmatize_head else "token"
    head = nc_converter.convert(head, tag=tag)

    tag = "lemma" if lemmatize_tail else "token"
    tail = nc_converter.convert(tail, tag=tag)

    tag = "lemma" if lemmatize_relations else "token"
    relation = " ".join([nc_converter.convert(i, tag=tag) for i in relation])

    # if head, tail or relation is invalid None will have been returned
    if (head is None) or (tail is None) or (relation is None):
        return None
    return head, relation, tail


def filter_triplets(relation_set: tuple, threshold: float,
                    continuous: bool = True):
    """
    relation_set (tuple): consist of a triplet and a confidence.
    The triplet (head, tail, relation), in the form of a path through it
    attention matrix from head through relation to tail
    continuous: checks if the relation is cont.

    this functions filters the follows
    1) relation should be an cont. sequence (to be implemented yet)
    2) confidence should be above threshold
    3) length of relation should be > 0
    """

    triplet = relation_set[0]
    confidence = relation_set[1]

    # check is relation is continuous
    if continuous and (not is_continous(triplet[1:-1])):
        return ()
    if confidence >= threshold and len(relations) > 0:
        return (triplet, confidence)
    return ()


def parse_sentence(
    tokens: list,
    noun_chunks: list,
    noun_chunk_token_span: list,
    lemmas: list,
    pos: list,
    ner: list,
    dependencies: list,
    attention,
    tokenizer,
    threshold: float,
    invalid_pos: set = {},
    invalid_dep: set = {},
    num_return_paths: int = 1,
    aggregate_method: str = "mult",
    n_beams: int = 6,
    max_length=None,
    min_length: int = 3,
    alpha: float = 1,
    lemmatize_relations: bool = False,
    filter_non_continous=True,
    lemmatize_head=False,
    lemmatize_tail=False

):
    """
    Example:
    >>> from utils import load_example
    >>> parse_sentence(**load_example(attention=True), threshold=0.005)
    """

    if len({len(tokens), len(lemmas), len(dependencies)}) > 1:
        raise ValueError(f"tokens, lemmas, ner, pos, and dependencies should have the same\
            length it is currently {len(tokens)}, {len(lemmas)}, {len(ner)}, \
                 {len(pos)}, and {len(dependencies)}, respectively")

    if len(noun_chunks) == 0:
        return []

    wordpiece2token_id, nc_tokens_id2tokens_id, noun_chunk_w_id = \
        create_mapping(tokens,
                       noun_chunks,
                       noun_chunk_token_span,
                       tokenizer)

    agg_attn = aggregate_attentions_heads(attention, head_dim=0)

    agg_attn = trim_attention_matrix(agg_attn,
                                     remove_padding=True,
                                     remove_eos=True,
                                     remove_bos=True)

    assert agg_attn.shape[0] == len(wordpiece2token_id), \
        "attention matrix and wordpiece2token does not have the same length"

    merged_attn = merge_token_attention(agg_attn, wordpiece2token_id)

    # make a forward and backward attention graph
    backward_attn_graph, forward_attn_graph = \
        attn_to_graph(merged_attn)

    # create head tail pair
    tail_head_pairs = []
    for head, h_idx in noun_chunk_w_id:
        for tail, t_idx in noun_chunk_w_id:
            if h_idx != t_idx:
                tail_head_pairs.append((h_idx, t_idx))

    # beam search
    def beam_search_(args):
        head, tail = args
        graph = forward_attn_graph if head < tail else backward_attn_graph
        return beam_search(head, tail,
                           graph=graph,
                           n_beams=n_beams,
                           alpha=alpha,
                           max_length=max_length,
                           min_length=min_length,
                           num_return_paths=num_return_paths,
                           aggregate_method=aggregate_method)

    all_relation_pairs = []
    for output in map(beam_search_, tail_head_pairs):
        if len(output):
            all_relation_pairs += output

    # filter
    nc_converter = NounChunkTokenIDConverter(nc_tokens_id2tokens_id, tokens,
                                             lemmas, pos, dependencies, ner)
    if invalid_pos:
        nc_converter.add_invalid(invalid_pos, tag="pos")
    if invalid_dep:
        nc_converter.add_invalid(invalid_dep, tag="dependencies")

    def __filter_to_str(relation_set):
        relation_set = filter_triplets(relation_set=relation_set,
                                       threshold=threshold,
                                       continuous=filter_non_continous)
        if not relation_set:
            return {}
        triplet, conf = relation_set
        triplet = triplet_to_str(triplet,
                                 nc_converter,
                                 lemmatize_relations=lemmatize_relations,
                                 lemmatize_head=lemmatize_head,
                                 lemmatize_tail=lemmatize_tail
                                 )
        if not triplet:
            return {}
        return {"head": triplet[0],
                "relation": triplet[1],
                "tail": triplet[2],
                "confidence": conf}

    triplets = []
    for triplet in map(__filter_to_str, all_relation_pairs):
        if triplet:
            triplets.append(triplet)
    return triplets


if __name__ == "__main__":
    from utils import load_example
    parse_sentence(**load_example(attention=True), threshold=0.005)
