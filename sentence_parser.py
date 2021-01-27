"""
"""

from functools import partial
import torch

from utils import create_mapping, build_graph, merge_token_attention, BFS
from utils import is_a_range as is_continous
from constants import invalid_relations_set


def bfs(args):
    s, end, graph, max_size, black_list_relation = args
    return BFS(s, end, graph, max_size, black_list_relation)


def aggregate_attentions_heads(
        attention, aggregate_fun=torch.mean, head_dim=1):
    """
    attention: all layers of attention from the model
    layer: the layer you wish to reduce by applying the aggregate_fun to
    aggregate_fun: the aggregation function
    head_dim: which dimension is the head dim which you want to aggregate over
    """
    return aggregate_fun(attention, dim=head_dim)


def filter_invalid_triplets(relation_set, id2token, id2tags, threshold):
    """
    relation_set (tuple): consist of a triplet and a confidence.
    The triplet (head, tail, relation), in the form of a path through it
    attention matrix


    1) confidence should be above threshold
    2) length of relation should be > 0
    3) relation should be an cont. sequence (to be implemented yet)
    3)
    """

    triplet_idx = relation_set[0]
    confidence = relation_set[1]
    head, tail = triplet_idx[0], triplet_idx[-1]

    assert head in id2token and (tail in id2token), \
        "head or tail not in id2token something must have gone wrong"

    head = id2token[head]
    tail = id2token[tail]

    # lemmatize relations
    relations = [id2tags[idx]["lemma"] for idx in triplet_idx[1:-1]]

    if len(relations) > 1:
        print("an example relation bigger than 1")  # should happen
    if not is_continous(triplet_idx[1:-1]):
        print("an example of a non cont. relation")

    if (confidence >= threshold and
            len(relations) > 0 and
            is_continous(triplet_idx[1:-1]) and
            check_relations_validity(relations) and
            head.lower() not in invalid_relations_set and
            (tail.lower() not in invalid_relations_set)):
        return {'head': head, 'tail': tail, 'relation': relations,
                'confidence': confidence}
    return {}


def check_relations_validity(relations):
    for rel in relations:
        if rel.lower() in invalid_relations_set or rel.isnumeric():
            return False
    return True


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
        threshold: float
):
"""
"""

    if len({len(tokens), len(lemmas), len(dependencies)}) > 1:
        raise ValueError(f"tokens, lemmas, ner, pos, and dependencies should have the same\
            length it is currently {len(tokens)}, {len(lemmas)}, {len(ner)}, \
                 {len(pos)}, and {len(dependencies)}, respectively")

    if len(noun_chunks) == 0:
        return []

    print(noun_chunks)

    tokenid2wordpiece, token2id, id2tags, noun_chunks = \
        create_mapping(tokens,
                       noun_chunks,
                       noun_chunk_token_span,
                       lemmas,
                       pos,
                       ner,
                       dependencies,
                       tokenizer)

    agg_attn = aggregate_attentions_heads(attention, head_dim=0)

    # fix size of attention matrix (remove padding)
    agg_attn = agg_attn[agg_attn.sum(dim=0) != 0, :]  # remove padding
    agg_attn = agg_attn[:, agg_attn.sum(dim=0) != 0]
    agg_attn = agg_attn[1: -1, 1: -1]  # remove eos and bos tokens

    assert agg_attn.shape[0] == len(tokenid2wordpiece), \
        "attention matrix and tokenid2wordpiece does not have the same length"

    merged_attn = merge_token_attention(agg_attn, tokenid2wordpiece)

    attn_graph = build_graph(merged_attn)

    # create head tail pair
    tail_head_pairs = []
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    # beam search
    black_list_relation = set([token2id[n] for n in noun_chunks])

    params = [(pair[0], pair[1], attn_graph, max(
        tokenid2wordpiece), black_list_relation) for pair in tail_head_pairs]

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
                              threshold=threshold)

    for triplet in map(filter_triplets, all_relation_pairs):
        if len(triplet):
            triplets.append(triplet)
    return triplets
