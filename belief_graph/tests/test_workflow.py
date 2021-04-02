"""
this is the intended workflow structure
"""
import sys
sys.path.append("/Users/au561649/Desktop/Github/UCLA-Conspiracies")

from collections.abc import Iterable

import belief_graph as bg

EXAMPLE = "Genåbningen sker nemlig på baggrund af et kontakttal for smitten på 1,0, hvilket betyder, at den lige nu ikke udvikler sig"

def test_belief_parser():
    nlp = bg.load_danish()
    belief_parser = bg.BeliefParser(nlp = nlp)
    relation_pairs = belief_parser.parse_texts(EXAMPLE)
    assert isinstance(relation_pairs, Iterable)
    bt = next(relation_pairs)
    assert isinstance(bt, bg.BeliefTriplet)

# test_belief_parser()

# def test_belief_graph():
#     belief_parser = bg.BeliefParser()
#     nlp =
#     graph = bg.BeliefGraph(parser=belief_parser, nlp=nlp)
#     graph.parse_texts()

#     relation_pairs = belief_parser.parse_sentence(EXAMPLE)
#     assert isinstance(relation_pairs, Iterable)
#     bt = next(relation_pairs)
#     assert isinstance(bt, bg.BeliefTriplet)
