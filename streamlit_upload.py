import os

import dill
import numpy as np
import streamlit as st
from streamlit_agraph import Config, Edge, Node, TripleStore, agraph

import belief_graph as bg
from belief_graph.tests.examples import EXAMPLES
from streamlit_utils import beliefgraph_to_triplestore, cache_on_button_press, pos_list

###########
## TODO
## - Add DepFilter
## - Make CountFilter work?
## - Make work with stored graph
## - Save nodes after creation of graph to see which nodes persist when changing filters
##########


apptitle = "KG Explorer"
st.set_page_config(page_title=apptitle, page_icon=":sunglasses:")

st.title("Knowledge Graph Explorer")

# Sidebar
filepath = st.sidebar.text_input("Enter a file path:")

st.sidebar.write(
    "### Choose filters. If the filter has options, an item will appear in the sidebar"
)

kg_filters = st.sidebar.multiselect(
    "Filters",
    [
        "ConfidenceFilter",
        "ContinuousFilter",
        "CountFilter",
        "DepFilter",  # Not implemented
        "EntFilter",
        "PosFilter",
        "LemmatizationFilter",
    ],
)

filters = {}
group_filters = {}

if "ConfidenceFilter" in kg_filters:
    expander = st.sidebar.beta_expander("ConfidenceFilter settings", expanded=True)
    with expander:
        conf_filter = st.slider("Threshold:", 0.0, 0.1, step=0.001, format="%0.3f")
    filters["ConfidenceFilter"] = bg.filters.ConfidenceFilter(threshold=conf_filter)

if "ContinuousFilter" in kg_filters:
    expander = st.sidebar.beta_expander("ContinuousFilter settings", expanded=True)
    with expander:
        conf_filter = st.checkbox("Activate", value=True)
    if conf_filter:
        filters["ContinuousFilter"] = bg.filters.ContinuousFilter()
    else:
        filters.pop("ContinuousFilter")

if "CountFilter" in kg_filters:
    expander = st.sidebar.beta_expander("CountFilter settings", expanded=True)
    with expander:
        count_filter = st.slider("Minimum count", 1, 100)  # might require some tuning
    group_filters["CountFilter"] = bg.filters.CountFilter(count=count_filter)

if "DepFilter" in kg_filters:
    pass

if "EntFilter" in kg_filters:
    expander = st.sidebar.beta_expander("EntFilter settings", expanded=True)
    with expander:
        ent_apply_to = st.multiselect("Apply to", ["head", "tail"])
        ent_valid = st.multiselect("Valid", [None, "LOC", "PER", "ORG", "MISC"])
        ent_invalid = st.multiselect("Invalid", [None, "LOC", "PER", "ORG", "MISC"])

    filters["EntFilter"] = bg.filters.EntFilter(
        valid=set(ent_valid), invalid=set(ent_invalid), apply_to=set(ent_apply_to)
    )

if "PosFilter" in kg_filters:
    expander = st.sidebar.beta_expander("PosFilter settings", expanded=True)
    with expander:
        pos_valid = st.multiselect("Valid POS", pos_list, key="pos_valid_key")
        pos_invalid = st.multiselect("Invalid POS", pos_list, key="pos_invalid_key")
        pos_apply_to = st.multiselect("Apply to", ["head", "tail"], key="pos_apply_key")
    filters["PosFilter"] = bg.filters.PosFilter(
        valid=set(pos_valid), invalid=set(pos_invalid), apply_to=set(pos_apply_to)
    )

if "LemmatizationFilter" in kg_filters:
    filters["LemmatizationFilter"] = bg.filters.LemmatizationFilter()


# Load model and cache
@cache_on_button_press("Load model", allow_output_mutation=True)
def load_model(path):
    graph = bg.BeliefGraph.from_disk(path)
    return graph


@cache_on_button_press("Load model and apply filters")
def load_and_add_filters(path, triplet_filters, group_filters):
    graph = bg.BeliefGraph.from_disk(path)
    graph.replace_filters(triplet_filters=triplet_filters, group_filters=group_filters)
    return graph


# Debug check to see if filters are added correctly
if st.button("Print input to terminal"):
    print(filters.values())
    print(group_filters.values())

# Set filters
graph = load_and_add_filters(
    filepath, list(filters.values()), list(group_filters.values())
)
# Get the nodes
nodes = set([triplet.head for triplet in graph.filtered_triplets])
# Choose nodes to plot
st.write("N. nodes loaded: " + str(len(nodes)))
heads = st.multiselect("Choose nodes to plot", list(nodes))

# convert selected nodes to streamlit graph store
store = beliefgraph_to_triplestore(graph, heads)

config = Config(
    height=600,
    width=700,
    nodeHighlightBehavior=True,
    highlightColor="#F7A7A6",
    directed=True,
    collapsible=True,
    link={"renderLabel": True},
)

agraph(list(store.getNodes()), (store.getEdges()), config)
