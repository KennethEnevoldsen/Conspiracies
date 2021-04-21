from .BeliefGraph import BeliefGraph
#import belief_graph as bg
#from belief_graph.tests.examples import EXAMPLES
from typing import Union, List

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


class BeliefNetwork:

    def __init__(self, graph: BeliefGraph):
        self.graph = graph
        if graph.is_filtered:
            self.triplets = list(graph.filtered_triplets)
        else:
            self.triplets = list(graph.triplets)

        self.labels = None
        self.weights = None
        self.G = nx.DiGraph()


    def construct_graph(self, 
                        scale_confidence: bool = True, 
                        nodes: Union[List[str], str] = "all"):
        """
        Construct entire KG for calculation of various measures
        such as centrality, degree, etc

        Params:
            scale_confidence: whether to weight edges by confidence
            nodes: which nodes to include. all relations, or 
            string/list of strings to specify specific relations
        """
        if nodes != "all":
            if not isinstance(nodes, (list, str, set)):
                raise TypeError(
                    f"Nodes should be a string or list of strings, not {type(nodes)}")
            if isinstance(nodes, list):
                nodes = set(nodes)
            if isinstance(nodes, str):
                nodes = {nodes}
                
        if scale_confidence:
            self.__add_weighted_nodes(nodes)
        else:
            self.__add_unweighted_nodes(nodes)

        # t[0] = head, t[1] = tail, t[2] = relation
        self.labels = {(t.head, t.tail): t.relation 
                       for t in self.triplets
                       if t.head in nodes}


    def plot_graph(self, save_name="none", **kwargs):
        if self.weights is None:
            raise ValueError("No graph has been constructed yet. Run construct_graph() first")
        
        degree = dict(self.G.degree)
        pos = nx.spring_layout(self.G)

        nx.draw(self.G, pos, edge_color=self.weights,
                alpha=0.8, width=self.weights,
                labels={node:node for node in self.G.nodes()},
                node_color=list(degree.values()),
                cmap=plt.cm.jet, verticalalignment="top",
                **kwargs)

        nx.draw_networkx(self.G, verticalalignment="bottom")
       # nx.draw_networkx_edges(self.G, pos)
       # nx.draw_networkx_edge_labels(self.G,pos,edge_labels=self.labels,
       #         font_color='red')
        
        
        plt.axis('off')
        if save_name != "none":
            plt.savefig(save_name + ".png", dpi=300)
  

    def __add_weighted_nodes(self, nodes):
        if nodes == "all":
            edges = [
                (triplet.head, triplet.tail, triplet.confidence) 
                for triplet in self.triplets
                    ]
            self.G.add_weighted_edges_from(edges)
        else:
            edges = [
                (triplet.head, triplet.tail, triplet.confidence) 
                for triplet in self.triplets
                if triplet.head in nodes
                    ]
            self.G.add_weighted_edges_from(edges)

        self.weights = list(nx.get_edge_attributes(self.G,'weight').values())

    def __add_unweighted_nodes(self, nodes):
        if nodes == "all":
            edges = [
                (triplet.head, triplet.tail) 
                for triplet in self.triplets
                ]
            self.G.add_edges_from(edges)
        else:
            edges = [
                (triplet.head, triplet.tail) 
                for triplet in self.triplets
                if triplet.head in nodes]
            self.G.add_edges_from(edges)

        self.weights = np.repeat(1, len(edges))


