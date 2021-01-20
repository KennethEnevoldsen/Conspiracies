import spacy

txt = "FBI om angreb på Kongressen: 'Jeg tror, folk bliver chokerede'"

nlp = spacy.load('da_core_news_lg', disable=["textcat"])


doc = nlp(txt)
for sent in doc.sents:
    for nc in sent.noun_chunks:
        print(nc.text)
    print("\n---")
