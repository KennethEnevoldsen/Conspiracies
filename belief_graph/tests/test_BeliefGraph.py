import belief_graph as bg
import pytest

from .examples import EXAMPLES
from .test_BeliefTriplet import simple_triplets


@pytest.fixture
def simple_graph():
    nlp = bg.load_danish()
    bp = bg.BeliefParser(nlp=nlp)

    graph = bg.BeliefGraph(parser=bp, triplet_filters=[], group_filters=[])
    return graph


def test_simple_graph(simple_graph):
    graph = simple_graph
    graph.add_texts(EXAMPLES)


def test_triplets(simple_graph):
    graph = simple_graph

    lt = len(graph.triplets)
    graph.add_texts(EXAMPLES)

    assert lt < len(graph.triplets)


def test_grouped_triplets(simple_graph):
    graph = simple_graph
    graph.add_texts(EXAMPLES)
    graph.add_texts(EXAMPLES)

    lg = len(graph.triplets)
    assert lg >= len(list(graph.triplet_groups))


def test_filtered_triplets(simple_graph):
    graph = simple_graph

    lf = len(list(graph.filtered_triplets))

    graph.add_texts(EXAMPLES)
    assert lf <= len(list(graph.filtered_triplets))
