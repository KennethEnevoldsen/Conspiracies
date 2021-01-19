"""
"""
from collections import defaultdict

import numpy as np
import torch

import transformers


def build_graph(matrix):
    """
    build a graph of top diagonal with format:
    idx: [(col, attention_value), ...]
    idx: [(col, attention_value), ...]
    ...

    Example:
    >>> mat = array([[10, 30, 30],
                     [20, 10, 30],
                     [20, 20, 10]])
    >>> build_graph(mat)
    defaultdict(list, {0: [(1, 30), (2, 30)], 1: [(2, 30)]})
    """
    graph = defaultdict(list)

    for idx in range(0, len(matrix)):
        for col in range(idx+1, len(matrix)):
            graph[idx].append((col, matrix[idx][col]))
    return graph


def create_mapping(spacy_dict: dict, return_pytorch=False,
                   tokenizer=transformers.AutoTokenizer.from_pretrained(
                       "Maltehb/-l-ctra-danish-electra-small-cased")):
    """
    Creates mapping from token to id, token to tokenizer id
    """

    tokens = spacy_dict["spacy_token"]

    start_chunk = [chunk[0]
                   for chunk in spacy_dict["spacy_noun_chunk_token_span"]]
    end_chunk = [chunk[1]
                 for chunk in spacy_dict["spacy_noun_chunk_token_span"]]
    noun_chunks = spacy_dict["spacy_noun_chunk"]

    sentence_mapping = []
    token2id = {}
    mode = 0  # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, token in enumerate(tokens):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token)
            token2id[sentence_mapping[-1]] = len(token2id)

    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token),
                                 add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [token2id[token]]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids +
                         [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pytorch:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(
                np.array(value)).long().unsqueeze(0)

    return outputs, tokenid2word_mapping, token2id, noun_chunks


def forward_pass(texts: list, tokenizer, model, device=None, **kwargs):
    """
    moves data to model device so model should be placed in the
    desired device

    >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
                       "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> res = forward_pass(["dette er en eksempel texts"], tokenizer, model)
    """
    if device is None:
        device = model.device

    with torch.no_grad():
        input_ = tokenizer(texts, return_tensors="pt", **kwargs)
        input_.to(device)
        output = model(**input_, output_attentions=True)

        # output[0].shape # batch, seq. length, embedding size
        res = {"attention": [t.to("cpu") for t in output.attentions],
            "embedding": output[0].to("cpu")}
    return res


def merge_token_attention(attention, tokenid2word, merge_operator=np.mean):
    """
    merge token attention to match spacy words
    """
    new_index = []

    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word[idx]
        if token_id != prev:
            new_index.append(row)
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(merge_operator(row.numpy(), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index = []
    for idx, row in enumerate(attention):
        token_id = tokenid2word[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(merge_operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    return new_matrix.T


def BFS(s, end, graph, max_size=-1, black_list_relation=[]):
    visited = [False] * (max(graph.keys())+100)

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append((s, [(s, 0)]))

    found_paths = []

    visited[s] = True

    while queue:

        s, path = queue.pop(0)

        # Get all adjacent vertices of the
        # dequeued vertex s. If a adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for i, conf in graph[s]:
            if i == end:
                found_paths.append(path+[(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path)+[(i, conf)]))
                visited[i] = True

    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf

        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))

    candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)
    return candidate_facts


# def dependency_relation_extractions(tokens, dependencies):
#     """
#     """
#     valid_relations = {("nsubj", "verb", "dobj"),
#                        ("nsubj", "verb", "(no obj)", "prep"),
#                        }
#     # constructed using loop
#     sub_valid_relations = {("nsubj")}

#     # for each relation

#     if relation in valid_relations:
#         yield relation
#     elif relation not in sub_valid_relations:
#         continue # take the next example
