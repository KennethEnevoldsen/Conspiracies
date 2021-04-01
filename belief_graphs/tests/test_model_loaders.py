"""
TODO: fix imports
"""

from model_loaders import load_danish

EXAMPLE = """
    Tirsdag kunne regeringen og støttepartierne præsentere en plan for den yderligere genåbning af Danmark.

    Og det er ikke alene godt nyt for de ældste skoleelever og store butikker. Der er også godt nyt til de bekymrede 
    danskere, der sidder og følger udviklingen i pandemien tæt. Genåbningen sker nemlig på baggrund af et kontakttal
    for smitten på 1,0, hvilket betyder, at den lige nu ikke udvikler sig.
    """


def test_danish():
    nlp = load_danish()

    doc = nlp(EXAMPLE0)

    for sent in doc.sents:
        print(sent)
        print("---")