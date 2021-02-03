"""
The script containing the SentenceParser class
"""
from typing import Optional, Union, List

from transformers import PreTrainedTokenizerBase


class SentenceParser():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_beams: int = 6,
        threshold: float = 0.005,
        lemmatize_relations: bool = False,
        invalid_pos: set = set(),
        invalid_dep: set = set(),
        max_length=None,
        min_length: int = 3,
        alpha: float = 1,
        num_return_paths: int = 1,
        aggregate_method: str = "mult",
    ):
        self.tokenizer = tokenizer
        self.n_beams = n_beams
        self.threshold = threshold
        self.lemmatize_relations = lemmatize_relations
        self.invalid_pos = invalid_pos
        self.invalid_dep = invalid_dep
        self.max_length = max_length
        self.min_length = min_length
        self.alpha = alpha
        self.num_return_paths = num_return_paths
        self.aggregate_method = aggregate_method

    def parse_sentence(
        self,
        tokens: Optional[List[str]] = None,
        noun_chunks: Optional[List[str]] = None,
        noun_chunk_token_span: Optional[List[Tuple[int, int]]] = None,
        lemmas: Optional[List[str]] = None,
        pos: Optional[List[str]] = None,
        ner: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ):
        pass

    def filter_sentence():
        pass

        