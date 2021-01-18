"""
"""

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "bert-base-uncased")
model = transformers.BertModel.from_pretrained(
    "bert-base-uncased")

input_ = tokenizer(["this is a sample sentence"], return_tensors="pt",
                   # add some padding
                   padding="max_length", max_length=128, truncation=True)
output = model(**input_)

# extract padding token embedding
pad_tok_id = [i for i, t in enumerate(input_["input_ids"][0]) if t == 0]
embedding_pad1 = output[0][0][pad_tok_id[0]]
embedding_pad2 = output[0][0][pad_tok_id[1]]

embedding_pad1.shape #embedding size
embedding_pad1[0:10]
embedding_pad2[0:10]