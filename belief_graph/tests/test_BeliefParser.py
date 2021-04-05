"""
this is the intended workflow structure
"""

from collections.abc import Iterable

import belief_graph as bg

EXAMPLE = "Genåbningen sker nemlig på baggrund af et kontakttal for smitten på 1,0, hvilket betyder, at den lige nu ikke udvikler sig"


def test_belief_parser():
    nlp = bg.load_danish()
    belief_parser = bg.BeliefParser(nlp=nlp)
    triplets = belief_parser.parse_texts(EXAMPLE)

    assert isinstance(triplets, Iterable)

    triplets = list(triplets)
    assert isinstance(triplets[0], bg.BeliefTriplet)

    for t in triplets:
        print(t)
