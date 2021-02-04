"""
Interviews
"""
import os
import ndjson
import csv

import torch

import pandas as pd
import numpy as np

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

from streamlit_network import load_data

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


def check_relations(path):
    df = load_data(path)
    df["accept"] = np.nan
    print("Press 'e' to accept, 'w' to reject")


    save_path = path[:-4] + "_thresholded.csv"
    if not os.path.exists(save_path):
        header = "count,confidence,sentence_number,document_id,head,relation,tail,accept\n"
        with open(save_path, 'w') as f:
            f.write(header)

    for idx, row in df.iterrows():
        proposition = f"h: {row['head']} r: {row['relation']} t: {row['tail']} "
        answer = input(proposition)
        row['accept'] = 1 if answer == "e" else 0

        with open(save_path, "a") as f:
            write = csv.writer(f)
            write.writerow(row.tolist())

    return None

if __name__ == '__main__':
    path = "results/2021-02-03-14.14_honest-markhor_threshold0.0.csv" 
    check_relations(path)
            
