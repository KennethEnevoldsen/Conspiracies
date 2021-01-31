"""
"""

from functools import partial

from utils import (
    create_wordpiece_token_mapping,
    matrix_to_graph,
    merge_token_attention,
    aggregate_attentions_heads,
    trim_attention_matrix,
    BFS
)
from utils import is_a_range as is_continous


def bfs(args):
    s, end, graph, max_size, black_list_relation = args
    return BFS(s, end, graph, max_size, black_list_relation)


def is_relation_valid(i: int, id2tags: dict,
                      invalid_pos={"NUM", "ADJ", "PUNCT", "ADV", "CCONJ",
                                   "CONJ", "PROPN", "NOUN", "PRON", "SYM"},
                      invalid_dep={}
                      ):
    """
    i: id of the token
    checks if a relation is valid
    """
    pos = id2tags[i]["pos"]
    dep = id2tags[i]["dependency"]

    # if list then it is a noun chunk
    if (isinstance(pos, list) or
            (pos in invalid_pos) or
            (dep in invalid_dep)):
        return False
    return True


def create_mapping(
        tokens,
        noun_chunks,
        noun_chunk_token_span,
        lemmas,
        pos,
        ner,
        dependencies,
        tokenizer):
    """
    tokenizer: a huggingface tokenizer
    Creates mappings from token id to its tokens as its tags.
    it also creates a mapping from a token to the tokenizer id

    Example:
    >>> mappings = create_mapping(**load_example(no_attention=True))
    """

    start_chunk = {s: e for s, e in noun_chunk_token_span}

    sentence_mapping = []
    token2id = {}
    id2tags = {}

    i = 0
    chunk_id = 0
    while i < len(tokens):
        id_ = len(token2id)
        if i in start_chunk:
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = id_
            chunk_id += 1
            end_chunk = start_chunk[i]
            id2tags[id_] = {"lemma": lemmas[i:end_chunk],
                            "pos": pos[i:end_chunk],
                            "ner": ner[i:end_chunk],
                            "dependency": dependencies[i:end_chunk]}
            i = end_chunk
        else:  # if not in chunk
            sentence_mapping.append(tokens[i])
            token2id[sentence_mapping[-1]] = id_
            id2tags[id_] = {"lemma": lemmas[i],
                            "pos": pos[i],
                            "ner": ner[i],
                            "dependency": dependencies[i]}
            i += 1

    wordpiece2token = create_wordpiece_token_mapping(
        sentence_mapping, token2id, tokenizer)

    return wordpiece2token, token2id, id2tags, noun_chunks


def filter_invalid_triplets(relation_set, id2token, id2tags, threshold,
                            invalid_pos, invalid_dep):
    """
    relation_set (tuple): consist of a triplet and a confidence.
    The triplet (head, tail, relation), in the form of a path through it
    attention matrix from head through relation to tail

    this functions filters the follows
    0) removed invalid pos and dependency-tag based on id2tags
    1) confidence should be above threshold
    2) length of relation should be > 0
    3) relation should be an cont. sequence (to be implemented yet)
    """

    triplet_idx = relation_set[0]
    confidence = relation_set[1]
    head, tail = triplet_idx[0], triplet_idx[-1]

    assert head in id2token and (tail in id2token), \
        "head or tail not in id2token something must have gone wrong"

    head = id2token[head]
    tail = id2token[tail]

    is_valid = partial(is_relation_valid, id2tags=id2tags,
                       invalid_pos=invalid_pos, invalid_dep=invalid_dep)

    # lemmatize relations and discard invalid relations
    relations = [id2tags[idx]["lemma"] for idx in triplet_idx[1:-1]
                 if is_valid(idx)]

    if len(relations) > 1:
        print("an example relation bigger than 1")  # should happen
    if not is_continous(triplet_idx[1:-1]):
        print("an example of a non cont. relation")

    if (confidence >= threshold and
            len(relations) > 0 and
            is_continous(triplet_idx[1:-1])):
        print("coo")
        return {'head': head, 'relation': relations, 'tail': tail,
                'confidence': confidence}
    return {}


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
    invalid_pos: set,
    invalid_dep: set
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

    print(noun_chunks)

    wordpiece2token, token2id, id2tags, noun_chunks = \
        create_mapping(tokens,
                       noun_chunks,
                       noun_chunk_token_span,
                       lemmas,
                       pos,
                       ner,
                       dependencies,
                       tokenizer)

    agg_attn = aggregate_attentions_heads(attention, head_dim=0)

    agg_attn = trim_attention_matrix(
        emove_padding=True, remove_eos=True, remove_bos=True)

    assert agg_attn.shape[0] == len(wordpiece2token), \
        "attention matrix and wordpiece2token does not have the same length"

    merged_attn = merge_token_attention(agg_attn, wordpiece2token)

    attn_graph = matrix_to_graph(merged_attn)

    # create head tail pair
    tail_head_pairs = []
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    # beam search
    black_list_relation = set([token2id[n] for n in noun_chunks])

    params = [(pair[0], pair[1], attn_graph, max(
        wordpiece2token), black_list_relation) for pair in tail_head_pairs]

    all_relation_pairs = []
    id2token = {value: key for key, value in token2id.items()}

    for output in map(bfs, params):
        if len(output):
            all_relation_pairs += output

    # filter
    triplets = []

    filter_triplets = partial(filter_invalid_triplets,
                              id2token=id2token,
                              id2tags=id2tags,
                              threshold=threshold,
                              invalid_pos=invalid_pos,
                              invalid_dep=invalid_dep)

    for triplet in map(filter_triplets, all_relation_pairs):
        if len(triplet):
            all_relation_pairs[0]
            triplets.append(triplet)
    return triplets
