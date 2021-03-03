"""
This script contains function for extracting/parsing beliefs (a
proposed knowledge triplet) from a text
"""

from transformers import PreTrainedTokenizerBase
from spacy.tokens import Doc

class BeliefParser():
    """
    A class for extracting belief triplets from a spacy doc
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_beams: int = 6,
        max_length=None,
        min_length: int = 3,
        alpha: float = 1,
        num_return_paths: int = 1,
        filter_non_continous: bool = True,
        aggregate_method: str = "mult",
    ):
        self.tokenizer = tokenizer
        self.n_beams = n_beams
        self.max_length = max_length
        self.min_length = min_length
        self.alpha = alpha
        self.num_return_paths = num_return_paths
        self.filter_non_continous = filter_non_continous
        self.aggregate_method = aggregate_method

    def parse_sentence(
        self,
        doc: Doc,
        attention,
    ):
        """
        doc: a SpaCy Doc or equivalent
        attention: an attention matrix from the forward pass of a transformer
        
        Example:
        >>> from utils import load_example
        >>> parse_sentence(**load_example(attention=True), threshold=0.005)
        """

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
