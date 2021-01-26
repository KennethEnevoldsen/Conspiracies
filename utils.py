"""
"""
from collections import defaultdict
from copy import copy

import numpy as np


def is_a_range(L):
    """
    checks if a list is equal to a range

    Examples:
    >>> is_a_range([2, 3, 4])
    True
    >>> is_a_range([2, 4, 5])
    False
    """
    for i, j in zip(range(L[0], L[-1]), L):
        if i != j:
            return False
    return True


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
    Creates mapping from token to id, token to tokenizer id

    Example:
    create_mapping()
    """

    zip_ = zip(tokens, lemmas, pos, ner, dependencies)

    start_chunk, end_chunk = zip(*noun_chunk_token_span)
    start_chunk, end_chunk = set(start_chunk), set(end_chunk)

    sentence_mapping = []
    token2id = {}
    id2tags = {}
    mode = 0  # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, z in enumerate(zip_):
        token, lemma, pos_, ner_, dep = z
        if idx in start_chunk:
            mode = 1
            id_ = len(token2id)

            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = id_
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token)

            id_ = len(token2id)
            token2id[sentence_mapping[-1]] = id_
            id2tags[id_] = {"lemma": lemma,
                            "pos": pos_,
                            "ner": ner_,
                            "dependency": dep}

    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token),
                                 add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [token2id[token]]*len(subtoken_ids)
        token_ids += subtoken_ids

    return tokenid2word_mapping, token2id, id2tags, noun_chunks


def merge_token_attention(attention, tokenid2word, merge_operator=np.mean):
    """
    merge token attention to match spacy words
    """
    new_index = []
    attention = attention.numpy()

    prev = -1
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
            if visited[i] is False:
                queue.append((i, copy(path)+[(i, conf)]))
                visited[i] = True

    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:  # if it only head and tail
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf

        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))

    return candidate_facts


def create_run_name(custom_name: str = None,
                    date: bool = True,
                    date_format: str = '%Y-%m-%d-%H.%M',
                    n_slugs: int = 2,
                    suffix: str = ""):
    """
    custom_name (str|None): custom name of the run, typically with a date
    added if it is none it will use the slug

    Example:
    >>> run_name = create_run_name(date=True, date_format="%Y-%m-%d", \
                                   n_slugs=2)
    >>> len(run_name.split("_")) == 2
    True
    >>> run_name = create_run_name(date=False, n_slugs=2)
    >>> len(run_name.split("-")) >= 2
    True
    """
    from coolname import generate_slug

    if custom_name is None:
        name = generate_slug(n_slugs)
    else:
        name = custom_name

    if date:
        from datetime import datetime
        name = datetime.today().strftime('%Y-%m-%d-%H.%M') + "_" + name

    name += suffix
    return name

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
