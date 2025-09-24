import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn import GraphConv, SAGEConv
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import (
    SchemaA1Dataset,
    FullFrenchTweetDataset,
    OldSchemaA1Dataset,
    SelectedDataset,
    ARENASFrenchAnnotator1Dataset,
    ARENASFrenchAnnotator2Dataset,
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
)

DATASETS = {
    "SchemaA1": SchemaA1Dataset,
    "FullFrenchTweet": FullFrenchTweetDataset,
    "OldSchemaA1": OldSchemaA1Dataset,
    "Selected": SelectedDataset,
    "ARENASFrenchAnnotator1": ARENASFrenchAnnotator1Dataset,
    "ARENASFrenchAnnotator2": ARENASFrenchAnnotator2Dataset,
    "Toxigen": ToxigenDataset,
    "LGBTEn": LGBTEnDataset,
    "MigrantsEn": MigrantsEnDataset,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Downstream node classification on arbitrary dataset and adjacency matrix")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help='Which dataset to use')
    parser.add_argument('--label_col', type=str, required=True,
                        help='Column in the CSV to use as the label for classification')
    parser.add_argument('--adjacency_matrix', type=str, required=True,
                        help='Path to the adjacency matrix pickle file')
    parser.add_argument('--experiment_nb', type=int, default=3,
                        help='Experiment number (for dataset init)')
    parser.add_argument('--embedding_col', type=str, default=None,
                        help='If your dataset supports multiple embedding columns, specify here')
    parser.add_argument('--train_split', type=float, default=0.4,
                        help='Ratio of nodes for training')
    parser.add_argument('--test_split', type=float, default=0.6,
                        help='Ratio of nodes for test')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--model', type=str, default='GraphSAGE', choices=['GCN', 'GraphSAGE'],
                        help='GNN model to use')
    parser.add_argument('--runs', type=int, default=10,
                        help='How many runs to average')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    return parser.parse_args()

def load_dataset(name, experiment_nb):
    return DATASETS[name](experiment_nb=experiment_nb)

def run_once(args, run_idx):
    # Set unique seed for reproducibility
    np.random.seed(args.seed + run_idx)
    torch.manual_seed(args.seed + run_idx)
    # Load adjacency matrix
    with open(args.adjacency_matrix, 'rb') as f:
        adj_matrix = pickle.load(f)
    if isinstance(adj_matrix, torch.Tensor):
        adj = adj_matrix.numpy()
    else:
        adj = adj_matrix

    # Load dataset
    dataset = load_dataset(args.dataset, args.experiment_nb)
    node_features = dataset.data

    if args.label_col not in node_features.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in dataset columns: {node_features.columns}")
    node_features = node_features.dropna(subset=[args.label_col])
    node_features = node_features.reset_index(drop=True)

    # Label encoding
    unique_labels = node_features[args.label_col].unique()
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    node_features['label'] = node_features[args.label_col].map(label_to_num)
    nb_classes = len(unique_labels)

    # Embeddings
    if args.embedding_col and args.embedding_col in node_features.columns:
        tweet_embeddings = node_features[args.embedding_col].values
    else:
        tweet_embeddings = dataset.embeddings
    if isinstance(tweet_embeddings, list):
        tweet_embeddings = np.stack(tweet_embeddings)
    elif isinstance(tweet_embeddings, pd.Series):
        tweet_embeddings = np.stack(tweet_embeddings.values)
    elif isinstance(tweet_embeddings, np.ndarray):
        pass
    else:
        tweet_embeddings = np.array(tweet_embeddings)

    assert adj.shape[0] == len(node_features), "Mismatch between adjacency matrix and node count"
    assert tweet_embeddings.shape[0] == len(node_features), "Mismatch between embeddings and node count"

    src, dst = np.nonzero(adj)
    graph = dgl.graph((src, dst))
    graph = dgl.to_simple(graph)

    labels_tensor = torch.tensor(node_features['label'].values, dtype=torch.long)
    graph.ndata['feat'] = torch.tensor(tweet_embeddings, dtype=torch.float32)
    graph.ndata['label'] = labels_tensor

    # Train/Test split (shuffle per run)
    n_nodes = graph.num_nodes()
    all_indices = np.arange(n_nodes)
    np.random.shuffle(all_indices)
    train_size = int(args.train_split * n_nodes)
    test_size = int(args.test_split * n_nodes)
    train_idx = all_indices[:train_size]
    test_idx = all_indices[train_size:train_size+test_size]

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['test_mask'] = test_mask

    # Model definition
    class GCN(nn.Module):
        def __init__(self, in_feats, hidden_feats, num_classes):
            super(GCN, self).__init__()
            self.layer1 = GraphConv(in_feats, hidden_feats)
            self.layer2 = GraphConv(hidden_feats, num_classes)
        def forward(self, graph, features):
            x = self.layer1(graph, features)
            x = F.relu(x)
            x = self.layer2(graph, x)
            return x

    class GraphSAGE(nn.Module):
        def __init__(self, in_feats, hidden_feats, num_classes, aggregator_type='mean', dropout=0.5):
            super(GraphSAGE, self).__init__()
            self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type)
            self.sage2 = SAGEConv(hidden_feats, num_classes, aggregator_type)
            self.dropout = nn.Dropout(dropout)
        def forward(self, graph, features):
            h = self.sage1(graph, features)
            h = F.relu(h)
            h = self.dropout(h)
            h = self.sage2(graph, h)
            return h

    in_feats = tweet_embeddings.shape[1]
    hidden_feats = 32
    learning_rate = 0.01
    num_epochs = args.epochs

    if args.model == "GCN":
        model = GCN(in_feats, hidden_feats, nb_classes)
    else:
        model = GraphSAGE(in_feats, hidden_feats, nb_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        logits = model(graph, graph.ndata['feat'])
        loss = loss_fn(logits[graph.ndata['train_mask']],
                       graph.ndata['label'][graph.ndata['train_mask']])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata['feat'])
        pred_class = logits.argmax(dim=1)
        test_pred = pred_class[graph.ndata['test_mask']].cpu().numpy()
        test_true = graph.ndata['label'][graph.ndata['test_mask']].cpu().numpy()
        acc = accuracy_score(test_true, test_pred)
        f1_macro = f1_score(test_true, test_pred, average="macro")
        f1_micro = f1_score(test_true, test_pred, average="micro")
        precision_macro = precision_score(test_true, test_pred, average="macro", zero_division=0)
        recall_macro = recall_score(test_true, test_pred, average="macro", zero_division=0)
        precision_micro = precision_score(test_true, test_pred, average="micro", zero_division=0)
        recall_micro = recall_score(test_true, test_pred, average="micro", zero_division=0)
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
        }

def main():
    args = parse_args()
    all_results = []
    for run_idx in range(args.runs):
        run_result = run_once(args, run_idx)
        print(f"\n=== Run {run_idx+1}/{args.runs} ===")
        print(f"Accuracy:        {run_result['accuracy']:.4f}")
        print(f"F1 Macro:        {run_result['f1_macro']:.4f}")
        print(f"F1 Micro:        {run_result['f1_micro']:.4f}")
        print(f"Precision Macro: {run_result['precision_macro']:.4f}")
        print(f"Recall Macro:    {run_result['recall_macro']:.4f}")
        print(f"Precision Micro: {run_result['precision_micro']:.4f}")
        print(f"Recall Micro:    {run_result['recall_micro']:.4f}")
        all_results.append(run_result)
    # Aggregate results
    metrics = list(all_results[0].keys())
    avg_results = {metric: np.mean([r[metric] for r in all_results]) for metric in metrics}
    std_results = {metric: np.std([r[metric] for r in all_results]) for metric in metrics}
    print("\n=== Average over {} runs ===".format(args.runs))
    for metric in metrics:
        print(f"{metric}: {avg_results[metric]:.4f} ± {std_results[metric]:.4f}")

if __name__ == "__main__":
    main()