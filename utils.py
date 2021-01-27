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


def load_example(attention=False):
    """
    laod an example for testing functions
    """
    import transformers
    import numpy as np
    import torch
    model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokens = ['På', 'de', 'seks', 'dage', ',', 'der', 'er', 'gået', 'siden',
              'angrebet', 'mod', 'Kongressen', ',', 'er', 'der', 'blevet',
              'rejst', 'over', '170', 'sager', 'mod', 'personer', ',', 'hvor',
              'flere', 'end', '70', 'allerede', 'er', 'blevet', 'sigtet', '.']
    noun_chunks = ['de seks dage', 'der', 'angrebet', 'Kongressen',
                   '170 sager', 'personer']
    noun_chunk_token_span = [[1, 4], [5, 6], [9, 10], [11, 12], [18, 20],
                             [21, 22]]
    lemmas = ['På', 'de', 'seks', 'dag', ',', 'der', 'være', 'gå', 'side',
              'angribe', 'mod', 'Kongressen', ',', 'være', 'der', 'blive',
              'rejse', 'over', '170', 'sag', 'mod', 'person', ',', 'hvor',
              'flere', 'ende', '70', 'allerede', 'være', 'blive', 'sigte',
              '.']
    pos = ['ADP', 'DET', 'NUM', 'NOUN', 'PUNCT', 'PRON', 'AUX', 'VERB', 'ADP',
           'NOUN', 'ADP', 'NOUN', 'PUNCT', 'AUX', 'ADV', 'AUX', 'VERB', 'ADP',
           'NUM', 'NOUN', 'ADP', 'NOUN', 'PUNCT', 'ADV', 'ADJ', 'ADP', 'NUM',
           'ADV', 'AUX', 'AUX', 'VERB', 'PUNCT']
    ner = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
           '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
    dependencies = ['case', 'det', 'nummod', 'obl', 'punct', 'nsubj', 'aux',
                    'acl:relcl', 'case', 'obl', 'case', 'nmod', 'punct', 'aux',
                    'expl', 'aux', 'ROOT', 'case', 'nummod', 'obl', 'case',
                    'nmod', 'punct', 'advmod', 'nsubj', 'case', 'nummod',
                    'advmod', 'aux', 'aux', 'acl:relcl', 'punct']
    example = {"tokenizer": tokenizer, "pos": pos, "ner": ner,
               "tokens": tokens,
               "dependencies": dependencies,
               "lemmas": lemmas,
               "noun_chunks": noun_chunks,
               "noun_chunk_token_span": noun_chunk_token_span}
    if attention:
        if attention is True:
            attention = np.load("example_attn.npy")
            attention = torch.Tensor(attention)
        example["attention"] = attention
    return example


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


def create_wordpiece_token_mapping(tokens: list, token2id: dict, tokenizer):
    """
    tokens: a list of tokens to map to tokenizer
    token2id: a mapping between token and its id
    tokenizer: a huggingface tokenizer

    create a mapping from wordpieces to tokens id in the form of a list
    e.g.
    [0, 1, 1, 1, 2]
    indicate that there are three tokens (0, 1, 2) and token 1 consist of three
    wordpieces
    note: this only works under the assumption that the word pieces
    are trained using similar tokens. (e.g. split by whitespace)
    """
    wordpiece2token = []
    for token in tokens:
        subtoken_ids = tokenizer(token,
                                 add_special_tokens=False)['input_ids']
        wordpiece2token += [token2id[token]]*len(subtoken_ids)
    return wordpiece2token


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
