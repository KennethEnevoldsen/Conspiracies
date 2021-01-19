"""
- [x] load dataset (and/or create)
- [x] apply spacy pipeline
- [x] apply forward pass using DaBERT (save embedding + attention)
- [ ] apply some kind of coref (not yet)
- [ ] extract knowledge graph
    - [x] create mapping betweeen bert and spacy
    - [x] extract head tail pair
    - [x] extract candidate facts
    - [ ] filter candidate facts
- [ ] add an argparse
- [ ] write dataset
    - [ ] add relevant metadata (tokenizer, model, spacy model)
"""
import os
from functools import partial

import torch

import transformers
import datasets
import spacy

from spacy_parsing import spacy_preprocess
from utils import forward_pass
from sentence_parser import parse_sentence


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
    preprocessed = spacy_preprocess(
        batch["text"], nlp=nlp, n_process=spacy_n_process)
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


def __tokenizer(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)


def forward_pass(texts: list, tokenizer, model, device=None, **kwargs):
    """
    moves data to model device so model should be placed in the
    desired device

    >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
                       "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> res = forward_pass(["dette er en eksempel texts"], tokenizer, model)
    """
    if device is None:
        device = model.device

    with torch.no_grad():
        input_ = tokenizer(texts, return_tensors="pt", **kwargs)
        input_.to(device)
        output = model(**input_, output_attentions=True)

        # output[0].shape # batch, seq. length, embedding size
        res = {"attention": [t.to("cpu") for t in output.attentions],
               "embedding": output[0].to("cpu")}
    return res


def batch(iterable, n=1):
    l_ = len(iterable)
    for ndx in range(0, l_, n):
        yield iterable[ndx:min(ndx + n, l_)]


if __name__ == '__main__':
    batch_size = None
    write_file = False
    model_parallel = True
    device = "cuda"
    attention_layer = -1
    spacy_n_process = 8

    # laod spacy pipeline
    nlp = spacy.load('da_core_news_lg', disable=["textcat"])
    # spacy.require_gpu()

    # load dataset using HF datasets
    ds = input_to_dataset()
    if batch_size is None:
        batch_size_ = len(ds)
    else:
        batch_size_ = batch_size
    ds = ds.map(__spacy_preprocess, batched=True, batch_size=batch_size_)

    # write preprocessed for other tasks
    if write_file:
        ds.set_format("pandas")
        df = ds[0:len(ds)]
        df.to_json("ds.ndjson", orient="records", lines=True)
        ds.reset_format()

    # turn file to sentences
    sent_ds = ds.map(doc_to_sent, batched=True, batch_size=batch_size_)

    # load tokenizers and transformer models
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    # enable GPU
    if model_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    ds = ds.map(__tokenizer, batched=True)
    ds.set_format(format="pt", columns=['attention_mask', 'input_ids'])

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors='pt')
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=collate_fn, batch_size=1024)

    # do forward pass
    with torch.no_grad():
        forward_batch = {"attention_weights": [], "embedding": []}

        for i, batch in enumerate(dataloader):
            batch.to(device)
            outputs = model(**batch, output_attentions=True)
            forward_batch["attention_weights"] = [a.to("cpu") for a in outputs.attentions]
            forward_batch["embedding"] = outputs[0]


    # extract KG
    parse_sentence_ = partial(
        parse_sentence, spacy_nlp=nlp, tokenizer=tokenizer)
    sent_ds = sent_ds.map(parse_sentence_)
    sent_ds
