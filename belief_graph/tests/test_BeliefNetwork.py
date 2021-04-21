import belief_graph as bg
import pytest

from .examples import EXAMPLES


@pytest.fixture
def simple_graph():
    nlp = bg.load_danish()
    bp = bg.BeliefParser(nlp=nlp)

    graph = bg.BeliefGraph(parser=bp, triplet_filters=[], group_filters=[])
    return graph

def create_simple_graph(simple_graph):
    graph = simple_graph
    graph.add_texts(EXAMPLES)
    return graph

def test_plot_all_nodes():
    s_g = create_simple_graph(simple_graph())
    bn = bg.BeliefNetwork(s_g)
    bn.construct_graph()
    bn.plot_graph()


def test_plot_specific_node():
    s_g = create_simple_graph(simple_graph())
    bn = bg.BeliefNetwork(s_g)
    bn.construct_graph(nodes="betyder", scale_confidence=False)
    bn.plot_graph()