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


def filter_relation_sets(params, spacy_nlp):
    triplet, id2token = params

    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
    if head in id2token and tail in id2token:
        head = id2token[head]
        tail = id2token[tail]
        relations = [spacy_nlp(id2token[idx])[
            0].lemma_ for idx in triplet_idx[1:-1] if idx in id2token]
        if (len(relations) > 0 and
                check_relations_validity(relations) and
                head.lower() not in invalid_relations_set and
                (tail.lower() not in invalid_relations_set)):
            return {'h': head, 't': tail, 'r': relations, 'c': confidence}
    return {}


def check_relations_validity(relations):
    for rel in relations:
        if rel.lower() in invalid_relations_set or rel.isnumeric():
            return False
    return True


def parse_sentence(spacy_dict, attention, tokenizer, spacy_nlp):
    """
    one or all sentence?
    """

    inputs, tokenid2word, token2id, noun_chunks = \
        create_mapping(spacy_dict, tokenizer=tokenizer)

    agg_attn = aggregate_attentions_heads(attention, head_dim=0)

    # fix size of attention matrix (remove padding)
    agg_attn = agg_attn[agg_attn.sum(dim=0) != 0, :]  # remove padding
    agg_attn = agg_attn[:, agg_attn.sum(dim=0) != 0]
    agg_attn = agg_attn[1:-1, 1:-1]  # remove eos and bos tokens

    assert agg_attn.shape[0] == len(
        tokenid2word), "attention matrix and tokenid2word does not have the same length"
    merged_attn = merge_token_attention(agg_attn, tokenid2word)
    # make graph
    attn_graph = build_graph(merged_attn)

    # head tail pair
    tail_head_pairs = []
    for head in noun_chunks:
        for tail in noun_chunks:
            if head != tail:
                tail_head_pairs.append((token2id[head], token2id[tail]))

    # beam search
    black_list_relation = set([token2id[n] for n in noun_chunks])

    all_relation_pairs = []
    id2token = {value: key for key, value in token2id.items()}

    params = [(pair[0], pair[1], attn_graph, max(
        tokenid2word), black_list_relation, ) for pair in tail_head_pairs]

    for output in map(bfs, params):
        if len(output):
            all_relation_pairs += [(o, id2token) for o in output]

    # filter
    triplet_text = []

    filter_relation_sets_ = partial(filter_relation_sets, spacy_nlp=spacy_nlp)

    for triplet in map(filter_relation_sets_, all_relation_pairs):
        if len(triplet) > 0:
            triplet_text.append(triplet)
    return triplet_text
