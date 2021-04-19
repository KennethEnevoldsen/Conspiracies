"""
This contains the expansion of the SpaCy Doc class to incorperate a forward
pass of the attention
"""

import spacy
from spacy.tokens import Doc
from spacy.tokens.span import Span

from .doc_extensions import (
    doc_nctokens_getter,
    doc_tokid2nc_getter,
    doc_tokid2ncid_getter,
    doc_tokid2wp_getter,
    doc_wp2ncid_getter,
    doc_wp2tokid_getter,
    span_attn_getter,
    span_nctokens_getter,
    span_wp2ncid_getter,
    span_wp2tokid_getter,
    span_wp_getter,
    span_wp_slice_getter,
)

Doc.set_extension("wp2ncid", getter=doc_wp2ncid_getter)
Doc.set_extension("nctokens", getter=doc_nctokens_getter)
Doc.set_extension("tokid2nc", getter=doc_tokid2nc_getter)
Doc.set_extension("wp2tokid", getter=doc_wp2tokid_getter)
Doc.set_extension("tokid2ncid", getter=doc_tokid2ncid_getter)
Doc.set_extension("tokid2wp", getter=doc_tokid2wp_getter)


Span.set_extension("wp_slice", getter=span_wp_slice_getter)
Span.set_extension("wp2tokid", getter=span_wp2tokid_getter)
Span.set_extension("attention", getter=span_attn_getter)
Span.set_extension("wordpieces", getter=span_wp_getter)
Span.set_extension("wp2ncid", getter=span_wp2ncid_getter)
Span.set_extension("nctokens", getter=span_nctokens_getter)



def load_danish(
    spacy_model: str = "da_core_news_sm", transformer: str = "Maltehb/danish-bert-botxo"
):
    nlp = spacy.load(spacy_model)

    if transformer:
        # add transformer
        # Construction via add_pipe with custom config
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v1",
                "name": transformer,
                "transformers_config": {"output_attentions": True},
                "tokenizer_config": {"use_fast": True, "strip_accents": False},
            }
        }
        trf = nlp.add_pipe("transformer", name="trf_forward", config=config)
        trf.model.initialize()

    return nlp
