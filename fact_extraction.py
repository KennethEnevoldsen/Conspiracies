"""
"""

import copy
import torch
import transformers
import datasets
import spacy

from spacy_parsing import spacy_preprocess

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Maltehb/-l-ctra-danish-electra-small-cased")
model = transformers.ElectraModel.from_pretrained(
    "Maltehb/-l-ctra-danish-electra-small-cased")
nlp = spacy.load('da_core_news_lg', disable=["textcat"])


def load_dataset(batch_size="max"):
    ds = datasets.load_dataset("dane", split="train")

    if batch_size == "max":
        batch_size = len(ds)

    def __tokenize(batch):
        return tokenizer(batch["text"])

    def __spacy_preprocess(batch):
        docs = nlp.pipe(batch["text"])
        preprocessed = spacy_preprocess(batch["text"], nlp=nlp, n_process=1)
        d = {}
        for doc in preprocessed:
            for key in doc.keys():
                skey = "spacy_" + key
                if skey not in d:
                    d[skey] = []
                d[skey].append(doc[key])
        return d

    ds = ds.map(__tokenize, batched=True, batch_size=batch_size)
    ds = ds.map(__spacy_preprocess, batched=True,
                batch_size=batch_size)
    return ds


if __name__ == '__main__':
    ds = load_dataset()
    ds_ = copy.copy(ds)

    ds_.set_format(type='torch',
                   columns=['input_ids',
                            'attention_mask'])
    out = model(**ds_[0:1], output_attentions=True)  # trying with two example
    type(out.attentions)
    len(out.attentions)
    ds[0:1]["input_ids"].shape
    out.attentions[11].shape

    out2 = model(**ds[1:2], output_attentions=True)


# input pipeline for text
# sentence splitting
    # noun chunk filtering


# ------ BEAMING -------- #
# implement beam search
# avg attention head
def reduce_attentions_heads(
        attention, layer: int = -1, aggregate_fun=torch.mean, head_dim=1):
    """
    attention: all layers of attention from the model
    layer: the layer you wish to reduce by applying the aggregate_fun to 
    aggregate_fun: the aggregation function
    head_dim: which dimension is the head dim which you want to aggregate over
    """
    return aggregate_fun(attention_matrices[layer], dim=head_dim)


# find head in attention (deal with multiple wordpieceses for one)

def create_mapping(sentence, nlp, tokenizer)

def aggregate_attention(
        attention, tokenid2wordmapping, aggregate_fun=torch.mean):
    """
    """
    pass

#
model.device
