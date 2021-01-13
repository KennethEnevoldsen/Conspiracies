"""
- [x] load dataset (and/or create)
- [x] apply spacy pipeline
- [x] apply forward pass using DaBERT (save embedding + attention)
- [ ] apply some kind of coref (not yet)
- [ ] extract knowledge graph
    - [x] create mapping betweeen bert and spacy
    - [ ] extract head tail pair
    - [ ] extract candidate facts
    - [ ] filter candidate facts
- [ ] add an argparse
- [ ] write dataset
    - [ ] add relevant metadata (tokenizer, model, spacy model)
"""
import os

import transformers
import datasets
import spacy

from spacy_parsing import spacy_preprocess
from utils import forward_pass


def input_to_dataset():
    def __gen_from_folder():
        for path in os.listdir("data"):
            with open(os.path.join("data", path), "r") as f:
                yield path, f.read()
    d = {"text": [], "filename": []}
    for fn, text in __gen_from_folder():
        d["text"].append(text)
        d["filename"].append(fn)

    return datasets.Dataset.from_dict(d)


def __spacy_preprocess(batch):
    preprocessed = spacy_preprocess(batch["text"], nlp=nlp)
    d = {}
    for doc in preprocessed:
        for key in doc.keys():
            skey = "spacy_" + key
            if skey not in d:
                d[skey] = []
            d[skey].append(doc[key])
    return d


def doc_to_sent(batch):
    """
    Split a preprocessed document into sentences
    """
    d = {}
    for key in batch.keys():
        d[key] = []

    for doc in range(len(batch["text"])):
        max_sent = max(batch["spacy_sent_id"][doc])
        i = 0
        start_idx = 0
        while i < max_sent:
            i += 1
            end_idx = batch["spacy_sent_id"][doc].index(i)

            for k in d.keys():
                # if in spacy add the sentence
                if k.startswith("spacy"):
                    d[k].append(batch[k][doc][start_idx:end_idx])
                # if in text add the sentence using idx
                elif k == "text":
                    start = batch["spacy_token_character_span"][doc][start_idx][0]
                    end = batch["spacy_token_character_span"][doc][end_idx - 1][1]
                    sent = batch[k][doc][start:end]
                    d[k].append(sent)
                # add all metadata to all derived sentences
                else:
                    d[k].append(batch[k][doc])
            start_idx = end_idx
    return d


def _forward_pass(batch):
    return forward_pass(
        batch["text"],
        model=model,
        tokenizer=tokenizer,
        padding="max_length", max_length=128, truncation=True)


if __name__ == '__main__':
    batch_size = None
    write_file = False
    # load tokenizers models and spacy pipe
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    nlp = spacy.load('da_core_news_lg', disable=["textcat"])

    # load dataset using HF datasets
    ds = input_to_dataset()
    if batch_size is None:
        batch_size = len(ds)
    ds = ds.map(__spacy_preprocess, batched=True, batch_size=batch_size)

    # write preprocessed for other tasks
    if write_file:
        pass
        # write file append to ndjson

    # turn file to sentences
    sent_ds = ds.map(doc_to_sent, batched=True, batch_size=batch_size)

    # apply forward pass
    if batch_size is None:
        batch_size = len(sent_ds)
    sent_ds = sent_ds.map(_forward_pass, batch_size=batch_size)

    # extract KG
