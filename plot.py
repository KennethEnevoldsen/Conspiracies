"""
Plot network graph
"""

import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter


def plot_graph(relations: list,
               scale_size: bool=True, 
               scale_confidence: bool=True,
               show_fig: bool=False,
               save_name: str="none"):
    """
    relations: list of dicts with keys "h", "t", "r", optionally "c"
    """
    G = nx.Graph()


    if scale_confidence:
        edges = [(rel["h"], rel["t"], rel["c"]) for rel in relations]
        G.add_weighted_edges_from(edges)

        weights = list(nx.get_edge_attributes(G,'weight').values())
        # Scale width by weight (experiment with scaling factor)
        weights = [w * 2 for w in weights]
    else:
        edges = [(rel["h"], rel["t"]) for rel in relations]
        G.add_edges_from(edges)

        weights = 1

    rels = [rel["r"] for rel in relations]

    labels = {(t[0], t[1]): r for t, r in zip(edges, rels)}

    pos=nx.spring_layout(G)

    if scale_size:
        degree = dict(G.degree)

        nx.draw(G,pos,edge_color='black',\
                node_color='pink',alpha=0.9,width=weights,\
                labels={node:node for node in G.nodes()},\
                node_size=[v * 500 for v in degree.values()])

    else:
        nx.draw(G,pos,edge_color='black',\
                node_size=100,node_color='pink',alpha=0.9,\
                labels={node:node for node in G.nodes()},
                width=weights)

    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,font_color='red')

    plt.axis('off')
    if save_name != "none":
        plt.savefig(save_name + ".png", dpi=300)
    
    if show_fig:
        plt.show()
    
    return None



if __name__ == '__main__':
    t_tup = [{"h": "Bob Dylan", "r": "loves", "t":"Ice cream", "c":0.3}, 
             {"h":"Bob Dylan", "r":"eats", "t":"dogs", "c":0.8},
             {"h":"John", "r":"is", "t":"a man", "c":1}]

    plot_graph(t_tup, save_name="test")

