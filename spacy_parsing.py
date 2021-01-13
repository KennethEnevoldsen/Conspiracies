"""

"""

import spacy
import transformers
import torch


def spacy_preprocess(texts: list,
                     n_process=1,
                     nlp=spacy.load('da_core_news_lg',
                                    disable=['textcat'])):
    '''
    '''
    docs = nlp.pipe(texts, n_process=n_process)

    def __extract_entity_noun_chunks(sent_dict):
        out = []
        for idx, e in enumerate(sent_dict["ner"]):
            if e:
                for c in sent_dict['noun_chunk']:
                    if sent_dict["text"][idx] in c:
                        out.append(c)
        return out

    spacy_tokens = []
    for doc in docs:
        doc_features = {}
        doc_features.update({
            #     'created_at': id,
            'text': [token.text for token in doc],
            'lemma': [token.lemma_ for token in doc],
            'pos': [token.pos_ for token in doc],
            'dep': [token.dep_ for token in doc],
            'ner': [token.ent_type_ for token in doc],
            'noun_chunk': [nc.text for nc in doc.noun_chunks]
        })
        doc_features['ent_noun_chunk'] = __extract_entity_noun_chunks(doc_features)

        spacy_tokens.append(doc_features)
    return spacy_tokens


def spacy_sentencizer(text,
                      nlp=spacy.load('da_core_news_lg',
                                     disable=['textcat'])):
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences



def create_mapping(sentence, return_pt=False, 
                   nlp=spacy.load('da_core_news_lg',
                                   disable=['textcat']), 
                   tokenizer=transformers.AutoTokenizer.from_pretrained(
                            "Maltehb/-l-ctra-danish-electra-small-cased")):
    '''Create a mapping
        nlp: spacy model
        tokenizer: huggingface tokenizer
    '''
    doc = nlp(sentence)

    tokens = list(doc)

    chunk2id = {}

    start_chunk = []
    end_chunk = []
    noun_chunks = []
    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)
        start_chunk.append(chunk.start)
        end_chunk.append(chunk.end)

    sentence_mapping = []
    token2id = {}
    mode = 0 # 1 in chunk, 0 not in chunk
    chunk_id = 0
    for idx, token in enumerate(doc):
        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(noun_chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(token2id)
            chunk_id += 1
        elif idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token.text)
            token2id[sentence_mapping[-1]] = len(token2id)

    token_ids = []
    tokenid2word_mapping = []

    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), 
                                 add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
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

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)
    
    return outputs, tokenid2word_mapping, token2id, noun_chunks







if __name__ == '__main__':
    spacy.prefer_gpu()
    # spacy.require_gpu()
    nlp = spacy.load('da_core_news_lg', disable=["textcat"])
    with open("data/dr_test_artikel.txt", "r") as f:
        text = f.read()
    sents = spacy_sentencizer(text, nlp)
    res = spacy_preprocess(texts=sents, nlp=nlp)
    res
    # list(res[0]["noun_chunk"])
