"""
The script containing the SentenceParser class as well as its related functions
"""
from typing import Optional, Union, List, Tuple

from transformers import PreTrainedTokenizerBase

from utils import (
    create_wordpiece_token_mapping,
    attn_to_graph,
    merge_token_attention,
    aggregate_attentions_heads,
    trim_attention_matrix,
    beam_search
)


def create_mapping(
        tokens,
        noun_chunks,
        noun_chunk_token_span,
        tokenizer):
    """
    tokenizer: a huggingface tokenizer
    Creates mappings from token id to its tokens as its tags.
    it also creates a mapping from a token to the tokenizer id

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

    # wordpiece2token: wordpiece -> Noun chunk merged tokens
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
            return self.convert_nc_to_str(nc_tok_id, tag)

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
            r = [self.tags[tag][i]
                 for i in range(tok_id[0], tok_id[1])
                 if self.is_tok_id_valid(i)]
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
    head = nc_converter.convert_nc_to_str(head, tag=tag)

    tag = "lemma" if lemmatize_tail else "token"
    tail = nc_converter.convert_nc_to_str(tail, tag=tag)

    tag = "lemma" if lemmatize_relations else "token"
    relation = [nc_converter.convert_to_str(i, tag=tag)
                for i in relation]
    relation = list(filter(lambda x: x, relation))
    if len(relation):
        relation = " ".join(relation)
    else:
        return None

    # if head, tail or relation is invalid None will have been returned
    if (head is None) or (tail is None):
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
    if confidence >= threshold and len(triplet[1:-1]) > 0:
        return (triplet, confidence)
    return ()


class ParseBelief():
    """
    A class for extracting belief triplets from a single sentence.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_beams: int = 6,
        threshold: float = 0.005,
        lemmatize_relations: bool = False,
        lemmatize_head: bool = False,
        lemmatize_tail: bool = False,
        invalid_pos: set = set(),
        invalid_dep: set = set(),
        max_length=None,
        min_length: int = 3,
        alpha: float = 1,
        num_return_paths: int = 1,
        filter_non_continous: bool = True,
        aggregate_method: str = "mult",
    ):
        self.tokenizer = tokenizer
        self.n_beams = n_beams
        self.threshold = threshold
        self.lemmatize_relations = lemmatize_relations
        self.lemmatize_head = lemmatize_head
        self.lemmatize_tail = lemmatize_tail
        self.invalid_pos = invalid_pos
        self.invalid_dep = invalid_dep
        self.max_length = max_length
        self.min_length = min_length
        self.alpha = alpha
        self.num_return_paths = num_return_paths
        self.filter_non_continous = filter_non_continous
        self.aggregate_method = aggregate_method

    def parse_sentence(
        self,
        tokens: List[str],
        noun_chunks: List[str],
        noun_chunk_token_span: List[Tuple[int, int]],
        attention,
        lemmas: Optional[List[str]] = None,
        pos: Optional[List[str]] = None,
        ner: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """
        It is expected that tokens, lemmas, pos, ner, dependencies have the
        same length

        Example:
        >>> from utils import load_example
        >>> parse_sentence(**load_example(attention=True), threshold=0.005)
        """
        self.tokens = tokens

        if len(noun_chunks) == 0:
            return []

        wordpiece2token_id, nc_tokens_id2tokens_id, noun_chunk_w_id = \
            create_mapping(tokens,
                           noun_chunks,
                           noun_chunk_token_span,
                           tokenizer=self.tokenizer)

        # create converter class for filtering
        self.nc_converter = NounChunkTokenIDConverter(
            nc_tokens_id2tokens_id, tokens, lemmas, pos, dependencies, ner)
        if self.invalid_pos:
            self.nc_converter.add_invalid(self.invalid_pos, tag="pos")
        if self.invalid_dep:
            self.nc_converter.add_invalid(self.invalid_dep, tag="dependencies")

        agg_attn = aggregate_attentions_heads(attention, head_dim=0)

        agg_attn = trim_attention_matrix(agg_attn,
                                         remove_padding=True,
                                         remove_eos=True,
                                         remove_bos=True)

        assert agg_attn.shape[0] == len(wordpiece2token_id), \
            "attention matrix and wordpiece2token does not have the same \
                length"

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
                               n_beams=self.n_beams,
                               alpha=self.alpha,
                               max_length=self.max_length,
                               min_length=self.min_length,
                               num_return_paths=self.num_return_paths,
                               aggregate_method=self.aggregate_method)

        self.relation_pairs = []
        for output in map(beam_search_, tail_head_pairs):
            if len(output):
                self.relation_pairs += output

    def filter_triplets(
        self,
        filter_non_continous: Optional[bool] = None,
        threshold: Optional[float] = None,
        lemmatize_relations: Optional[bool] = None,
        lemmatize_head: Optional[bool] = None,
        lemmatize_tail: Optional[bool] = None,
    ):
        """
        """
        if threshold is None:
            threshold == self.threshold
        if filter_non_continous is None:
            filter_non_continous == self.filter_non_continous
        if lemmatize_relations is None:
            lemmatize_relations == self.lemmatize_relations
        if lemmatize_head is None:
            lemmatize_head == self.lemmatize_head
        if lemmatize_tail is None:
            lemmatize_tail == self.lemmatize_tail

        # filter

        def __filter_to_str(relation_set):
            relation_set = filter_triplets(relation_set=relation_set,
                                           threshold=self.threshold,
                                           continuous=filter_non_continous)
            if not relation_set:
                return {}
            triplet, conf = relation_set
            triplet = triplet_to_str(triplet,
                                     self.nc_converter,
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

        self.triplets = []
        for triplet in map(__filter_to_str, self.relation_pairs):
            if triplet:
                self.triplets.append(triplet)
        return self.triplets




# Belief extract
    # init with params such as number of beams
    # takes a sentence with tags (noun chunks etc.)
    # extract triplets pr. sentence
        # return a list of BeliefTriplets

# BeliefTriplet
    # contains
        # triplets
        # conf
        # Nc converter (Noun Chunk to everything)


# Belief Graphs
    # init
        # takes a belief extraction function which return belief triplets
    # method for adding singular sentences (pass to belief extraction)
    # method for adding multiple sentences
    # (Method for unlisting belief triplets) 
    # method for filtering belief graphs
        # custom threshold etc.
    # plotting
    # app'en