from typing import Callable, List, Generator
import numpy as np
from numpy import ndarray
from collections import defaultdict
from .data_classes import BeliefTriplet, TripletGroup


def merge_token_attention(attention, tokenid2word, merge_operator=np.mean):
    """
    merge token attention to match spacy words
    """
    new_index = []

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


def beam_search(
    head: int,
    tail: int,
    graph: dict,
    n_beams: int = 6,
    alpha: float = 0,
    max_length=None,
    min_length: int = 3,
    num_return_paths=1,
    aggregate_method="mult",
):
    """
    head: the start of the search
    tail: the desired end of the search
    graph: a graph (network) to search through
    n_beams: the number of beam to use. If None implements a BFS (breatdth
    first search)
    alpha (0<alpha<1): The length normalization. The sum of p's (entries in
    the graph) is multiplied by a normalization constant which is 1/n where n
    is the number p's (i.e. length of the sequence). If alpha==1 this is the
    mean if alpha==0 there is no length normalization.
    max_length: max length of a path, stop beam if path length prematurely
    if path is longer than max length
    min_length: minimum length of a path
    (typically path less than 3 is irrelevant as it is only head and tail)
    num_return_paths (int|None): return best path if None returns all found
    paths
    aggregate_method ("mult"|"sum"): the method in which the weight should be
    aggregated if sum the weight are summed and then normalized otherwise they
    are multiplied (using log addition) and then normalized

    this function is implemented in a BFS style fashion.

    Example:
    matrix = np.array([[0.07, 0.27, 0.3 , 0.76, 0.01],
                        [0.24, 0.39, 0.14, 0.57, 0.16],
                        [0.12, 0.11, 0.14, 0.43, 0.13],
                        [0.66, 0.13, 0.48, 1.  , 0.48],
                        [0.48, 0.32, 0.37, 0.23, 0.59]])
    graph = matrix_to_graph(matrix)
    beam_search(head=0, tail=4, graph=graph, n_beams=2, alpha=1,
                num_return_paths=None)

    """
    visited = set()

    # Create a queue for BFS
    queue = []
    queue.append((head, [(head, 0)]))

    found_paths = []

    # starting search
    while queue:
        head_, path = queue.pop(0)
        node_sorted = sorted(graph[head_], key=lambda x: x[1], reverse=True)

        for node, conf in node_sorted[0:n_beams]:
            if node == tail:
                path_ = path + [(node, conf)]
                # disregard path if too short
                if min_length and len(path_) >= min_length:
                    found_paths.append(path_)
            else:
                # stop beam prematurely if length to long
                if max_length and len(path) >= max_length - 1:
                    continue
                if node not in visited:
                    queue.append((node, path + [(node, conf)]))
                    visited.add(node)

    candidate_facts = aggregate_and_normalize(found_paths, alpha)
    candidate_facts = sorted(candidate_facts, key=lambda x: x[1])

    if num_return_paths and num_return_paths >= len(candidate_facts):
        num_return_paths = len(candidate_facts) - 1

    return candidate_facts[0:num_return_paths]


def aggregate_and_normalize(found_paths, alpha, aggregate_method="mult"):
    candidate_facts = []
    for fp in found_paths:
        path, conf = zip(*fp)

        # aggregate
        if aggregate_method == "mult":
            conf = np.log(conf[1:])
            agg_conf = np.exp(conf.sum())
        elif aggregate_method == "sum":
            conf = conf[1:]
            agg_conf = conf.sum()

        # length normalize
        norm_conf = agg_conf * 1 / len(conf) ** alpha

        candidate_facts.append((path, norm_conf))
    return candidate_facts


def attn_to_graph(matrix):
    """
    build a forward (buttom diagonal) and backward (upper diagonal)
    graph with format:
    idx: [(col, attention_value), ...]
    idx: [(col, attention_value), ...]
    ...

    Example:
    >>> mat = np.array([[10, 30, 30],
                     [20, 10, 30],
                     [20, 20, 10]])
    >>> attn_to_graph(mat)
    (defaultdict(list, {0: [(1, 30), (2, 30)], 1: [(2, 30)]}),
    defaultdict(list, {2: [(0, 20), (1, 20)], 1: [(0, 20)]}))
    """
    backward_graph = defaultdict(list)
    for idx in reversed(range(0, len(matrix))):
        for col in range(0, idx):
            backward_graph[idx].append((col, matrix[idx][col]))

    forward_graph = defaultdict(list)
    for idx in range(0, len(matrix)):
        for col in range(idx + 1, len(matrix)):
            forward_graph[idx].append((col, matrix[idx][col]))

    return backward_graph, forward_graph




def merge_triplets(sorted_triplets: List[BeliefTriplet]) -> Generator:
    """
    sorted_triplets (List[BeliefTriplet]): Assumed a list of sorted triplets.
    """
    queue = sorted_triplets

    while queue:
        triplet = queue.pop()

        tg = TripletGroup.from_belief_triplet(triplet)
        while triplet == queue[-1]:
            tg.add(queue.pop())
        yield tg