import os

import belief_graph as bg


nlp = bg.load_danish(spacy_model="da_core_news_lg")
bp = bg.BeliefParser(nlp=nlp)

graph = bg.BeliefGraph(parser=bp, offload_dir="twitter_docs")

BASE_DIR = "data/dagw-master/sektioner/twfv19"
SAVE_NAME = "twitter_dump"

files = os.listdir(BASE_DIR)
files = list(filter(lambda x: x.startswith("twfv19"), files))


def load_text(files):
    for file in files:
        with open(os.path.join(BASE_DIR, file)) as f:
            yield f.read()


texts = load_text(files)

graph.add_texts(texts)

graph.to_disk(SAVE_NAME)
