"""
This contains the expansion of the SpaCy Doc class to incorperate a forward
pass of the attention
"""

import spacy
from spacy.tokens import Doc
from spacy.tokens.span import Span

from doc_extensions import doc_wp2tokid_getter, doc_tokid2wp_getter, span_wp2tokid_getter, span_attn_getter


Doc.set_extension("wp2tokid", getter=doc_wp2tokid_getter)
Doc.set_extension("tokid2wp", getter=doc_tokid2wp_getter)
Span.set_extension("wp2tokid", getter=span_wp2tokid_getter)
Span.set_extension("attention", getter=span_attn_getter)

def load_danish(
    spacy_model="da_core_news_sm", transformer="Maltehb/danish-bert-botxo", lemmy=False
):
    nlp = spacy.load("da_core_news_sm")

    if lemmy is True:
        import lemmy.pipe

        pipe = lemmy.pipe.load("da")
        nlp.add_pipe(pipe, after="tagger")

    if transformer:
        # add transformer
        # Construction via add_pipe with custom config
        config = {
            "model": {
                "@architectures": "spacy-transformers.TransformerModel.v1",
                "name": "Maltehb/danish-bert-botxo",
                "transformers_config": {"output_attentions": True},
            }
        }
        trf = nlp.add_pipe("transformer", name="trf_forward", config=config)
        trf.model.initialize()

    return nlp

if __name__ == "__main__":
    nlp = load_danish()

    doc = nlp(
        """Tirsdag kunne regeringen og støttepartierne præsentere en plan for den yderligere genåbning af Danmark.

    Og det er ikke alene godt nyt for de ældste skoleelever og store butikker. Der er også godt nyt til de bekymrede 
    danskere, der sidder og følger udviklingen i pandemien tæt. Genåbningen sker nemlig på baggrund af et kontakttal
    for smitten på 1,0, hvilket betyder, at den lige nu ikke udvikler sig.
    """
    )

    # docs = nlp.pipe(["this is a test", "test 2"])
    # d1, d2 = [doc for doc in docs]
    # d1._.trf_data.attention[-1].shape
    # print(d1._.trf_data.attention[-1][0][-1])
    # print(d2._.trf_data.attention[-1][0][-1])

    sents = []
    for i, sent in enumerate(doc.sents):
        sents.append(sent)

        print(sent)
        print("-")
        for nc in sent.noun_chunks:
            print(nc)
        print("---")

    sent_span = sents[0]

    sent_span._.attention.shape

