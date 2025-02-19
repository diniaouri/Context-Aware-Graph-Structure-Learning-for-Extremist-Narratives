import warnings
import pickle as pkl
import sys
import os
from preprocessing import GraphDataset
import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

warnings.simplefilter("ignore")


def load_data():
    return GraphDataset(adjacency_matrix_type="knn").get_dataset()
