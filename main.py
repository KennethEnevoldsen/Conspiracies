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


def forward_pass(ds, model, batch_size=1024):
    """
    does the forward pass on the dataset
    """
    model.eval()
    if model_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    ds = ds.map(__tokenizer, batched=True)
    ds.set_format(format="pt", columns=['attention_mask', 'input_ids'])

    # on the fly padding

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors='pt')
    dataloader = torch.utils.data.DataLoader(
        ds, collate_fn=collate_fn, batch_size=batch_size)

    # do forward pass
    forward_batches = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch.to(device)
            outputs = model(**batch, output_attentions=True)
            forward_batch = {
                "attention_weights": [a.to("cpu") for a in outputs.attentions],
                "embedding": outputs[0].to("cpu")}
            forward_batches.append(forward_batch)
    ds.reset_format()
    return ds, forward_batches


def unwrap_attention_from_batch(forward_batches, attention_layer=-1):
    for fb in forward_batches:
        attn = fb["attention_weights"][attention_layer]
        b_size = attn.shape[0]
        for i in range(b_size):
            yield attn[i]


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

    sent_ds, forward_batches = forward_pass(ds, model)
    attn = unwrap_attention_from_batch(forward_batches)

    # extract KG
    results = []
    for spacy_dict, attn in zip(sent_ds, attn):
        i = 10
        forward_batches[0]["embedding"].shape
        res = parse_sentence(spacy_dict=spacy_dict, attention=attn,
                             tokenizer=tokenizer, spacy_nlp=nlp)
        results.append(res)
