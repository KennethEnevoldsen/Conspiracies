"""
- [x] load dataset (and/or create)
- [ ] apply spacy pipeline
- [ ] apply forward pass using DaBERT (save embedding + attention)
- [ ] apply some kind of coref (not yet)
- [ ] extract knowledge graph
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


def input_to_dataset():
    def __gen_from_folder():
        for path in os.listdir("data"):
            with open(os.path.join("data", path), "r") as f:
                yield path, f.read()
    d = {"text": [], "filename": []}
    for fn, text in __gen_from_folder():
        d["text"].append(text)
        d["filename"].append(fn)

    return datasets.Dataset.from_dict(ds)


if __name__ == '__main__':
    # load tokenizers models and spacy pipe
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    nlp = spacy.load('da_core_news_lg', disable=["textcat"])

    # load dataset using HF datasets
    ds = input_to_dataset()

    # apply 
