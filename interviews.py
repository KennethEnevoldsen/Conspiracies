"""
Interviews
"""
import os
import ndjson

import torch

import pandas as pd

import transformers
import datasets
import spacy

from main import (parse_sentence_wrapper,
                  input_to_dataset,
                  __spacy_preprocess,
                  doc_to_sent,
                  __tokenizer,
                  forward_pass,
                  unwrap_attention_from_batch,
                  relation_count_filter)


def prepare_interviews(data_folder="interviews"):
    """
    Concats transcript from interviewees and saves as individual  txt files
    """
    df = pd.read_excel(os.path.join(data_folder, "AUVAC_data.xlsx"), engine="openpyxl")
    df  = df.loc[df["speaker"] == "interviewee"]

    texts = df.groupby(['id'])['text'].apply(lambda x: ' '.join(x)).reset_index()

    folder = "interview_data"
    if not os.path.exists(folder):
        os.makedirs(folder)


    for i in df["id"].unique():
        sub = texts.loc[texts["id"] == i]['text'].reset_index(drop=True)[0]
        fname = f"{folder}/interview_{i}.txt"
        with open(fname, "w") as f:
            f.write(sub)
    return None
