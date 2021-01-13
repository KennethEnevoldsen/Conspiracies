import datasets  # huggingface's datasets

import spacy

# load sample danish dataset
ds = datasets.load_dataset("dane", split="train")

nlp = spacy.load('da_core_news_lg')
docs = nlp.pipe(ds["text"])

len(ds)

max([len(list(doc.noun_chunks)) for doc in docs])