"""
"""
import os
import ndjson

import torch

import transformers
import datasets
import spacy

from spacy_parsing import spacy_preprocess
from utils import create_run_name
from sentence_parser import parse_sentence


def parse_sentence_wrapper(
        spacy_dict: dict,
        attention,
        tokenizer,
        threshold: float):
    print(spacy_dict["text"])

    triplet = parse_sentence(
        tokens=spacy_dict["spacy_token"],
        lemmas=spacy_dict["spacy_lemma"],
        ner=spacy_dict["spacy_ner"],
        pos=spacy_dict["spacy_pos"],
        dependencies=spacy_dict["spacy_dep"],
        noun_chunk_token_span=spacy_dict["spacy_noun_chunk_token_span"],
        noun_chunks=spacy_dict["spacy_noun_chunk"],
        attention=attention,
        tokenizer=tokenizer,
        threshold=threshold,
        invalid_pos=invalid_pos,
        invalid_dep=invalid_dep

    )
    return triplet


def input_to_dataset(folder="data"):
    def __gen_from_folder():
        for path in os.listdir(folder):
            with open(os.path.join(folder, path), "r") as f:
                yield path, f.read()
    d = {"text": [], "document_id": []}
    for fn, text in __gen_from_folder():
        d["text"].append(text)
        d["document_id"].append(fn)

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
    d = {"sent_id": []}
    for key in batch.keys():
        d[key] = []

    for doc in range(len(batch["text"])):
        max_sent = max(batch["spacy_sent_id"][doc])
        i = 0
        start_idx = 0
        start_nc = 0
        while i < max_sent:
            i += 1
            end_idx = batch["spacy_sent_id"][doc].index(i)

            # add noun_chunks to sent
            # identify sentence belonging to noun chunk
            nc_span = batch["spacy_noun_chunk_token_span"][doc]
            idx = start_nc
            while True:
                span = nc_span[idx]
                if span[0] >= end_idx:
                    end_nc = idx
                    break
                idx += 1

            # extract and append noun chunk
            nc_tok_span = nc_span[start_nc: end_nc]
            # set span to match new sentence (rather than doc)
            nc_tok_span = [[span[0]-start_idx, span[1]-start_idx]
                           for span in nc_tok_span]
            nc = batch["spacy_noun_chunk"][doc][start_nc: end_nc]
            d["spacy_noun_chunk_token_span"].append(nc_tok_span)
            d["spacy_noun_chunk"].append(nc)
            start_nc = end_nc

            for k in d.keys():
                if k.startswith("spacy_noun_chunk"):
                    continue
                # if in spacy add the sentence
                elif k.startswith("spacy"):
                    d[k].append(batch[k][doc][start_idx:end_idx])
                # if in text add the sentence using idx
                elif k == "text":
                    token_span = batch["spacy_token_character_span"][doc]
                    start = token_span[start_idx][0]
                    end = token_span[end_idx - 1][1]
                    sent = batch[k][doc][start:end]
                    d[k].append(sent)
                # add all metadata to all derived sentences
                elif k == "sent_id":
                    d[k].append(i)
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


def relation_count_filter(results, n: int):
    """
    filter relations based on count. Removed all relations
    with a count less than n
    """
    counter = {}
    for result in results:
        triplets = result["triplets"]
        sent_id = result["sentence_number"]
        document_id = result["document_id"]

        for triplet in triplets:
            key = (triplet["head"], triplet["relation"], triplet["tail"])
            if key in counter:
                counter[key]["count"] += 1
                counter[key]["confidence"] += [triplet["confidence"]]
                counter[key]["sentence_number"] += [sent_id]
                counter[key]["document_id"] += [document_id]
            else:
                counter[key]["count"] = 1
                counter[key]["confidence"] = [triplet["confidence"]]
                counter[key]["sentence_number"] = [sent_id]
                counter[key]["document_id"] = [document_id]

    for k in counter.keys():
        output = {}
        if counter[k]["count"] < n:
            continue
        output = counter[k]
        output["head"] = k[0]
        output["relation"] = k[1]
        output["tail"] = k[2]
        yield output


if __name__ == '__main__':
    batch_size = None
    write_file = False
    model_parallel = True
    attention_layer = -1
    spacy_n_process = 1
    device = None
    model_name = "Maltehb/-l-ctra-danish-electra-small-cased"
    threshold = 0.005
    min_count = 2
    invalid_pos = {"NUM", "ADJ", "PUNCT", "ADV", "CCONJ",
                   "CONJ", "PROPN", "NOUN", "PRON", "SYM"},
    invalid_dep = {}
    # confidence threshold is 0.003 in the public example and 0.005 in the
    # paper
    save_results = False

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        ds.save_to_disk("preprocessed")

    # turn file to sentences
    sent_ds = ds.map(doc_to_sent, batched=True, batch_size=batch_size_)

    # load tokenizers and transformer models
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.ElectraModel.from_pretrained(model_name)

    sent_ds, forward_batches = forward_pass(sent_ds, model)
    attentions = unwrap_attention_from_batch(forward_batches)

    # extract KG
    results = []
    for spacy_dict, attn in zip(sent_ds, attentions):
        triplets = parse_sentence_wrapper(spacy_dict=spacy_dict,
                                          attention=attn,
                                          tokenizer=tokenizer,
                                          threshold=threshold)

        results.append({"triplets": triplets,
                        "sentence_number": spacy_dict["sent_id"],
                        "document_id": spacy_dict["document_id"]})

    results = list(relation_count_filter(results, n=min_count))

    if save_results:
        params = f"_threshold{threshold}"
        save_path = os.path.join("results",
                                 create_run_name(suffix=params+".ndjson"))
        with open(save_path, "w") as f:
            ndjson.dump(results, f)
