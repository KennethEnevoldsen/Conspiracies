"""
This contains the expansion of the SpaCy Doc class to incorperate a forward
pass of the attention
"""

import spacy


def load_danish(spacy_model="da_core_news_sm", transformer="Maltehb/danish-bert-botxo", lemmy=True):
    nlp = spacy.load("da_core_news_sm")

    if lemmy is True:
        import lemmy.pipe
        pipe = lemmy.pipe.load('da')
        nlp.add_pipe(pipe, after='tagger')

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
        trf = nlp.add_pipe("trf_forward", config=config)
        trf.model.initialize()

    return nlp
