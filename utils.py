"""
"""

import transformers
import numpy as np


def create_mapping(spacy_dict: dict, return_pytorch=False,
                   tokenizer=transformers.AutoTokenizer.from_pretrained(
                       "Maltehb/-l-ctra-danish-electra-small-cased")):
    """
    Creates mapping from token to id, token to tokenizer id
    """

    tokens = spacy_dict["token"]

    chunk2id = {}

    start_chunk = [chunk[0] for chunk in spacy_dict["noun_chunk_token_span"]]
    end_chunk = [chunk[1] for chunk in spacy_dict["noun_chunk_token_span"]]
    noun_chunks = spacy_dict["noun_chunk"]

    sentence_mapping = []
    token2id = {}
    mode = 0  # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, token in enumerate(tokens):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token)
            token2id[sentence_mapping[-1]] = len(token2id)

    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token),
                                 add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [token2id[token]]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pytorch:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(
                np.array(value)).long().unsqueeze(0)

    return outputs, tokenid2word_mapping, token2id, noun_chunks


def forward_pass(texts: list, tokenizer, model, **kwargs):
    """
    moves data to model device so model should be placed in the
    desired device

    >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
                       "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> model = transformers.ElectraModel.from_pretrained(
        "Maltehb/-l-ctra-danish-electra-small-cased")
    >>> res = forward_pass(["dette er en eksempel texts"], tokenizer, model)
    """
    input_ = tokenizer(texts, return_tensors="pt", **kwargs)
    input_.to(model.device)
    output = model(**input_, output_attentions=True)
    # output[0].shape # batch, seq. length, embedding size
    return {"attention": output.attentions[-1],
            "embedding": output[0]}


def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []

    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index = []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append([row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    return new_matrix.T
