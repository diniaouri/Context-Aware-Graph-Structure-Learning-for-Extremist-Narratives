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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import (
    SchemaA1Dataset,
    FullFrenchTweetDataset,
    OldSchemaA1Dataset,
    SelectedDataset,
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

DATASETS = {
    "SchemaA1": SchemaA1Dataset,
    "FullFrenchTweet": FullFrenchTweetDataset,
    "OldSchemaA1": OldSchemaA1Dataset,
    "Selected": SelectedDataset,
    "ARENASFrenchAnnotator1": ARENASFrenchAnnotator1Dataset,
    "ARENASFrenchAnnotator2": ARENASFrenchAnnotator2Dataset,
    "ARENASGermanAnnotator1": ARENASGermanAnnotator1Dataset,
    "ARENASGermanAnnotator2": ARENASGermanAnnotator2Dataset,
    "ARENASCypriotAnnotator1": ARENASCypriotAnnotator1Dataset,
    "ARENASCypriotAnnotator2": ARENASCypriotAnnotator2Dataset,
    "ARENASSloveneAnnotator1": ARENASSloveneAnnotator1Dataset,
    "ARENASSloveneAnnotator2": ARENASSloveneAnnotator2Dataset,
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
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of stratified folds')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--model', type=str, default='GraphSAGE', choices=['GCN', 'GraphSAGE'],
                        help='GNN model to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of repeated runs (outer repetitions of k-fold)')
    return parser.parse_args()

def load_dataset(name, experiment_nb):
    return DATASETS[name](experiment_nb=experiment_nb)

def run_fold(graph, train_idx, test_idx, model_name, nb_classes, in_feats, tweet_embeddings, labels_tensor, epochs, device):
    train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['test_mask'] = test_mask

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

    hidden_feats = 32
    learning_rate = 0.01

    if model_name == "GCN":
        model = GCN(in_feats, hidden_feats, nb_classes)
    else:
        model = GraphSAGE(in_feats, hidden_feats, nb_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    features = torch.tensor(tweet_embeddings, dtype=torch.float32).to(device)
    labels = labels_tensor.to(device)
    graph = graph.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        logits = model(graph, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        pred_class = logits.argmax(dim=1)
        test_pred = pred_class[test_mask].cpu().numpy()
        test_true = labels[test_mask].cpu().numpy()
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

    all_runs_results = []
    per_run_avgs = []

    for run in range(args.n_runs):
        run_seed = args.seed + run
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

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
        in_feats = tweet_embeddings.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=run_seed)
        labels = node_features['label'].values
        run_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros_like(labels), labels)):
            print(f"\n=== Run {run+1}/{args.n_runs}, Fold {fold + 1}/{args.n_splits} ===")
            result = run_fold(
                graph, train_idx, test_idx, args.model, nb_classes,
                in_feats, tweet_embeddings, labels_tensor, args.epochs, device)
            for metric, val in result.items():
                print(f"{metric}: {val:.4f}")
            run_results.append(result)
        all_runs_results.append(run_results)

        # Per-run averages
        metrics = list(run_results[0].keys())
        run_avg = {metric: np.mean([fold_result[metric] for fold_result in run_results]) for metric in metrics}
        per_run_avgs.append(run_avg)
        print(f"\n=== Average for Run {run+1}/{args.n_runs} ===")
        for metric in metrics:
            vals = [fold_result[metric] for fold_result in run_results]
            print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Aggregate results across all runs and folds
    metrics = list(all_runs_results[0][0].keys())
    # Flatten all [runs][folds] into [runs*folds]
    all_metrics = {metric: [] for metric in metrics}
    for run_results in all_runs_results:
        for fold_result in run_results:
            for metric in metrics:
                all_metrics[metric].append(fold_result[metric])
    print("\n=== Overall Results Across All Runs and Folds ===")
    for metric in metrics:
        avg = np.mean(all_metrics[metric])
        std = np.std(all_metrics[metric])
        print(f"{metric}: {avg:.4f} ± {std:.4f}")

    # Averages of the 10 runs
    print("\n=== Average of the average of the {} runs ===".format(args.n_runs))
    for metric in metrics:
        per_run_metric_avgs = [run_avg[metric] for run_avg in per_run_avgs]
        avg = np.mean(per_run_metric_avgs)
        std = np.std(per_run_metric_avgs)
        print(f"{metric}: {avg:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()