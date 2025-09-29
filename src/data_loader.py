import warnings
import pickle as pkl
import sys
import os
from preprocessing import (
    GraphDataset, 
    SchemaA1Dataset, 
    FullFrenchTweetDataset, 
    ARENASFrenchAnnotator1Dataset, 
    ARENASFrenchAnnotator2Dataset,
    ARENASGermanAnnotator1Dataset,
    ARENASGermanAnnotator2Dataset,
    ARENASCypriotAnnotator1Dataset,
    ARENASCypriotAnnotator2Dataset,
    ARENASSloveneAnnotator1Dataset,
    ARENASSloveneAnnotator2Dataset,
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
)
import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

def load_data(dataset_name="graph", csv_path=None, experiment_nb=1):
    """
    Load data for the specified dataset.
    If dataset_name is "graph", uses GraphDataset.
    For others, uses the respective dataset class.
    Now supports ToxigenDataset for experiment 9.
    Also supports LGBTEnDataset and MigrantsEnDataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "graph":
        return GraphDataset(adjacency_matrix_type="knn").get_dataset()
    elif dataset_name == "schemaa1":
        if csv_path is None:
            raise ValueError("csv_path must be provided for SchemaA1Dataset.")
        return SchemaA1Dataset(csv_path)
    elif dataset_name == "fullfrenchtweet":
        return FullFrenchTweetDataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenasfrenchannotator1":
        return ARENASFrenchAnnotator1Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenasfrenchannotator2":
        return ARENASFrenchAnnotator2Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenasgermanannotator1":
        return ARENASGermanAnnotator1Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenasgermanannotator2":
        return ARENASGermanAnnotator2Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenascypriotannotator1":
        return ARENASCypriotAnnotator1Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenascypriotannotator2":
        return ARENASCypriotAnnotator2Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenassloveneannotator1":
        return ARENASSloveneAnnotator1Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "arenassloveneannotator2":
        return ARENASSloveneAnnotator2Dataset(experiment_nb=experiment_nb, skip_embeddings=True)
    elif dataset_name == "toxigen":
        return ToxigenDataset(experiment_nb=experiment_nb, csv_path=csv_path, skip_embeddings=True)
    elif dataset_name == "lgbten":
        return LGBTEnDataset(csv_path=csv_path)
    elif dataset_name == "migrantsen":
        return MigrantsEnDataset(csv_path=csv_path)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

# The below class is kept for legacy/simple usage; usually you should use the main class in preprocessing.py
class SchemaA1Dataset:
    """
    Lightweight CSV-based SchemaA1Dataset for legacy/simple usage.
    Now supports arbitrary context columns for get_context_attributes.
    """

    DEFAULT_CONTEXT_COLUMNS = [
        "Topic", "In_group", "Out_group", "Initiating Problem",
        "Intolerance", "Superiority of in-group",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Setting", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.df = df
        self.texts = df["Text"].tolist()
        # Legacy attributes, keep for backwards compatibility
        self.topics = df["Topic"].tolist() if "Topic" in df.columns else [None]*len(df)
        self.in_groups = df["In_group"].tolist() if "In_group" in df.columns else [None]*len(df)
        self.out_groups = df["Out_group"].tolist() if "Out_group" in df.columns else [None]*len(df)
        self.Initiating_Problem = df["Initiating Problem"].tolist() if "Initiating Problem" in df.columns else [None]*len(df)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "Text": self.texts[idx],
            "Topic": self.topics[idx],
            "In_group": self.in_groups[idx],
            "Out_group": self.out_groups[idx],
            "Initiating Problem": self.Initiating_Problem[idx]
        }

    def get_context_attributes(self, indices, columns=None):
        """
        Return list of tuples for context attributes for given indices.
        If columns is None, use all supported columns.
        """
        if columns is None:
            columns = self.DEFAULT_CONTEXT_COLUMNS
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.df.iloc[i][col] if col in self.df.columns else None)
            out.append(tuple(entry))
        return out 