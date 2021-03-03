"""
streamlit network app
"""
import ndjson
import ast

import pandas as pd
import numpy as np

import streamlit as st
from streamlit_agraph import agraph, TripleStore, Node, Edge, Config
#from layout import footer

path = "results/2021-02-03-14.14_honest-markhor_threshold0.0.csv"


def load_data(path):
    df = pd.read_csv(path)
    df["confidence"] = df["confidence"].apply(lambda x: ast.literal_eval(x))
    # Getting mean of the confidence scores
    df["confidence"] = df["confidence"].apply(lambda x: np.mean(x))
    return df


def subset_data(df, heads, threshold):
    if isinstance(heads, str):
        heads = [heads]
    data = df.copy()
    data = data.loc[(data["head"].isin(heads))
                    & (data["confidence"] > threshold), ]
    store = TripleStore()
    for idx, rel in data.iterrows():
        store.add_triple(rel["head"], rel["relation"], rel["tail"])

    return store


def app():
    df = load_data(path)
    st.title("Open Knowledge Graph Visualization")

    heads = st.multiselect('Choose heads', df["head"].unique())
    threshold = st.slider('Set confidence threshold',
                          0., df["confidence"].max(), value=0.0000001)

    store = subset_data(df, heads, threshold)

    config = Config(height=600, width=700, nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6", directed=True,
                    collapsible=True, link={"renderLabel": True})

    st.write("Nodes loaded: " + str(len(store.getNodes())))

    agraph(list(store.getNodes()), (store.getEdges()), config)


if __name__ == '__main__':
    path = "results/2021-02-03-14.14_honest-markhor_threshold0.0.csv"
    app()
