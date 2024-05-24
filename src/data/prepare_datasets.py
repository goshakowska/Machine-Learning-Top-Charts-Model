import pandas as pd


def get_dataframe(dataset: str):
    DATA_DIR = "../data/raw/Z04_T69_V2/"
    return pd.read_json(DATA_DIR + dataset + ".jsonl", lines=True)
