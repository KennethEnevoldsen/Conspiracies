"""
utility script for parsing sentence for belief graphs
"""
from collections import defaultdict

import os
import numpy as np
import torch


def is_a_range(L):
    """
    checks if a list is equal to a range

    Examples:
    >>> is_a_range([2, 3, 4])
    True
    >>> is_a_range([2, 4, 5])
    False
    >>> is_a_range(L=[1, 3, 2])
    False
    """
    L_ = range(L[0], L[-1]+1)
    if len(L_) != len(L):
        return False
    for i, j in zip(L_, L):
        if i != j:
            return False
    return True


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
        for col in range(idx+1, len(matrix)):
            forward_graph[idx].append((col, matrix[idx][col]))

    return backward_graph, forward_graph


def load_dict_in_memory(d: dict):
    """
    loads a dictionary into memory.

    a utility function
    load_dict_in_memory(load_example(True, True))
    """

    for key, item in d.items():
        exec("global " + key + "; " + key + " = item")


def load_example(attention=False, add_beam_params=False):
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
    invalid_pos = {"NUM", "ADJ", "PUNCT", "ADV", "CCONJ",
                   "CONJ", "PROPN", "NOUN", "PRON", "SYM"},
    invalid_dep = {}
    example = {"tokenizer": tokenizer, "pos": pos, "ner": ner,
               "tokens": tokens,
               "dependencies": dependencies,
               "lemmas": lemmas,
               "noun_chunks": noun_chunks,
               "noun_chunk_token_span": noun_chunk_token_span,
               "invalid_pos": invalid_pos,
               "invalid_dep": invalid_dep}
    if attention:
        if attention is True:
            attention = np.load("example_attn.npy")
            attention = torch.Tensor(attention)
        example["attention"] = attention
    if add_beam_params:
        example["alpha"] = 1
        example["n_beams"] = 3
        example["num_return_paths"] = 1
        example["aggregate_method"] = "mult"
        example["max_length"] = None
        example["min_length"] = 3
    return example


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


def aggregate_attentions_heads(
        attention, aggregate_fun=torch.mean, head_dim=1):
    """
    attention: all layers of attention from the model
    layer: the layer you wish to reduce by applying the aggregate_fun to
    aggregate_fun: the aggregation function
    head_dim: which dimension is the head dim which you want to aggregate over
    """
    return aggregate_fun(attention, dim=head_dim)


def trim_attention_matrix(agg_attn,
                          remove_padding: bool = True,
                          remove_eos: bool = True,
                          remove_bos: bool = True):
    """
    trim attention matrix by removing eos, bos and padding
    """
    if remove_padding:
        agg_attn = agg_attn[agg_attn.sum(dim=0) != 0, :]
        agg_attn = agg_attn[:, agg_attn.sum(dim=0) != 0]
    start_idx = 1 if remove_eos else 0

    if remove_eos:
        return agg_attn[start_idx: -1, start_idx: -1]
    else:
        return agg_attn[start_idx:, start_idx:]


def beam_search(head: int,
                tail: int,
                graph: dict,
                n_beams: int = 6,
                alpha: float = 0,
                max_length=None,
                min_length: int = 3,
                num_return_paths=1,
                aggregate_method="mult"):
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
        print("\t"*(len(path)-1), head_, end=" ")
        node_sorted = sorted(graph[head_], key=lambda x: x[1], reverse=True)

        print("going to: ", [i[0] for i in node_sorted[0:n_beams]])
        for node, conf in node_sorted[0:n_beams]:
            if node == tail:
                print("added to paths")
                path_ = path + [(node, conf)]
                # disregard path if too short
                if min_length and len(path_) >= min_length:
                    found_paths.append(path_)
            else:
                # stop beam prematurely if length to long
                if max_length and len(path) >= max_length - 1:
                    continue
                if node not in visited:
                    queue.append((node, path+[(node, conf)]))
                    visited.add(node)

    candidate_facts = aggregate_and_normalize(found_paths, alpha)
    candidate_facts = sorted(candidate_facts, key=lambda x: x[1])

    if num_return_paths and num_return_paths >= len(candidate_facts):
        num_return_paths = len(candidate_facts)-1

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
        norm_conf = agg_conf * 1/len(conf)**alpha

        candidate_facts.append((path, norm_conf))
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


def plot_network(relations_csv: str,
                 filename: str,
                 n_edges: int = 0):
    """
    Plot network with visNetwork
    keep n_edges < ~ 150-200
    """
    os.system(
        "Rscript --vanilla plot_network.R -f " +
        f"{relations_csv} -n {filename} -e {n_edges}")
    print(f"Network graph saved to {filename}")
    return None
