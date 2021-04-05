import belief_graph as bg

from .examples import EXAMPLES
from .test_BeliefTriplet import simple_triplets


def simple_graph(EXAMPLES):
    nlp = bg.load_danish()
    bp = bg.BeliefParser(nlp=nlp)


    tf = bg.filters.ContinuousFilter()

    graph = bg.BeliefGraph(parser=bp, triplet_filter=[tf], group_filter=[])
    graph.add(EXAMPLES)
    return graph


def test_triplets(simple_graph, EXAMPLES):
    graph = simple_graph

    lt = len(graph.triplets)

    graph.add(EXAMPLES)
    assert lt < len(graph.triplets)


def test_grouped_triplets(simple_graph):
    graph = simple_graph

    lg = len(graph.grouped_triplets)

    graph.add(EXAMPLES)
    assert lg == len(graph.grouped_triplets)


def test_filtered_triplets(simple_graph):
    graph = simple_graph

    lf = len(graph.filtered_triplets)

    graph.add(EXAMPLES)
    assert lf <= len(graph.filtered_triplets)


def test_add_text(EXAMPLES):
    nlp = bg.load_danish()
    bp = bg.BeliefParser(nlp=nlp)

    graph = bg.BeliefGraph(parser=bp)
    graph.add_text(EXAMPLES)


def test_add_triplet(simple_triplet):
    nlp = bg.load_danish()
    bp = bg.BeliefParser(nlp=nlp)

    graph = bg.BeliefGraph(parser=bp)
    graph.add_triplet(simple_triplet)
