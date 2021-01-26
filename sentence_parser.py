"""
"""

from functools import partial
import torch

from utils import create_mapping, build_graph, merge_token_attention, BFS
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


def filter_relation_sets(params, threshold):
    triplet, id2token, id2tags = params

    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]

    assert head in id2token and (tail in id2token), \
        "head or tail not in id2token something must have gone wrong"

    head = id2token[head]
    tail = id2token[tail]

    # lemmatize relation set
    for idx in triplet_idx[1:-1]:
        if idx not in id2tags:
            raise Exception("thsi should not be the case")
    relations = [id2tags[idx]["lemma"] for idx in triplet_idx[1:-1]]

    if (len(relations) > 0 and
            confidence >= threshold and
            check_relations_validity(relations) and
            head.lower() not in invalid_relations_set and
            (tail.lower() not in invalid_relations_set)):
        return {'h': head, 't': tail, 'r': relations, 'c': confidence}
    else:
        raise Exception("Head or tail not in id2token.\
             Please check if a bug is present")
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
    agg_attn = agg_attn[1:-1, 1:-1]  # remove eos and bos tokens

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
            all_relation_pairs += [(o, id2token, id2tags) for o in output]

    # filter
    triplet_text = []

    filter_relation_sets_ = partial(
        filter_relation_sets, threshold=threshold)

    for triplet in map(filter_relation_sets_, all_relation_pairs):
        if len(triplet) > 0:
            triplet_text.append(triplet)
    return triplet_text
