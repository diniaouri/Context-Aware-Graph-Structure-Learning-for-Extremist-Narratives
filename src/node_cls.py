#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GCNConv, BatchNorm, JumpingKnowledge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import random
from typing import Optional

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
    parser = argparse.ArgumentParser(
        description="Node classification with PyG (JK + BatchNorm), fixed split, multi-seed."
    )
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--adjacency_matrix', type=str, required=True)
    parser.add_argument('--experiment_nb', type=int, default=3)

    # Features (optional)
    parser.add_argument('--embeddings_file', type=str, default=None)
    parser.add_argument('--feature_cols', type=str, nargs='*', default=None)

    # Model
    parser.add_argument('--model', type=str, default='GraphSAGE', choices=['GCN', 'GraphSAGE'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--jk_mode', type=str, default='cat', choices=['cat', 'max', 'lstm', 'sum'])

    # Training
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--seeds', type=int, nargs='*', default=[0, 21, 42, 84, 123])
    parser.add_argument('--cpu_only', action='store_true')
    return parser.parse_args()

class GNNWithBNJK(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, model='GraphSAGE',
                 num_layers=3, dropout=0.2, jk_mode='cat'):
        super().__init__()
        conv_class = GCNConv if model == 'GCN' else SAGEConv
        self.convs = nn.ModuleList([conv_class(in_feats, hidden_dim)])
        self.norms = nn.ModuleList([BatchNorm(hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(conv_class(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.jk = JumpingKnowledge(jk_mode)
        out_feats = hidden_dim * num_layers if jk_mode == 'cat' else hidden_dim
        self.lin = nn.Linear(out_feats, num_classes)
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jk(xs)
        x = self.lin(x)
        return x

def build_features(node_df: pd.DataFrame, remaining_idx: np.ndarray,
                   embeddings_arr: Optional[np.ndarray], feature_cols: Optional[list]) -> Optional[np.ndarray]:
    feats = []

    if embeddings_arr is not None:
        emb_arr = np.asarray(embeddings_arr)
        feats.append(emb_arr[remaining_idx])

    if feature_cols:
        for col in feature_cols:
            col_data = node_df[col]
            if col_data.dtype == object or col_data.dtype.name == "category":
                feats.append(pd.get_dummies(col_data).values.astype(np.float32))
            else:
                feats.append(col_data.values.astype(np.float32).reshape(-1, 1))

    if not feats:
        return None
    return np.concatenate(feats, axis=1).astype(np.float32)

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # Load adjacency matrix
    with open(args.adjacency_matrix, 'rb') as f:
        adj_matrix = pickle.load(f)
    adj = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix

    # Load dataset
    dataset = DATASETS[args.dataset](experiment_nb=args.experiment_nb, embeddings_path=args.embeddings_file)
    df = dataset.data

    required_cols = [args.label_col] + (args.feature_cols or [])
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Encode labels
    unique_labels = df[args.label_col].unique()
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    df['label'] = df[args.label_col].map(label_to_num)
    labels = df['label'].values
    nb_classes = len(np.unique(labels))

    # Filter adjacency to remaining nodes
    remaining_idx = df.index.values
    adj = adj[np.ix_(remaining_idx, remaining_idx)]

    # Load features
    embeddings_arr = None
    if args.embeddings_file:
        embeddings_arr = np.load(args.embeddings_file)
    elif hasattr(dataset, 'embeddings') and dataset.embeddings is not None:
        embeddings_arr = dataset.embeddings

    features_np = build_features(df.set_index(np.arange(len(df))), np.arange(len(df)),
                                 embeddings_arr, args.feature_cols)
    if features_np is None:
        print("No features provided; using degree-based scalar feature.")
        features_np = adj.sum(axis=1).reshape(-1, 1).astype(np.float32)

    features = torch.tensor(features_np, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

    # Create **fixed train/val/test split**
    N = features.shape[0]
    idx = np.arange(N)
    np.random.seed(42)  # fixed split seed
    np.random.shuffle(idx)
    train_idx = torch.tensor(idx[:int(0.6 * N)], dtype=torch.long)
    val_idx   = torch.tensor(idx[int(0.6 * N):int(0.8 * N)], dtype=torch.long)
    test_idx  = torch.tensor(idx[int(0.8 * N):], dtype=torch.long)

    train_mask = torch.zeros(N, dtype=torch.bool, device=device); train_mask[train_idx] = True
    val_mask   = torch.zeros(N, dtype=torch.bool, device=device); val_mask[val_idx] = True
    test_mask  = torch.zeros(N, dtype=torch.bool, device=device); test_mask[test_idx] = True

    all_results = []

    for run_idx, seed in enumerate(args.seeds):
        print(f"\n=== Run {run_idx + 1}/{len(args.seeds)} | Seed={seed} ===")
        set_all_seeds(seed)

        model = GNNWithBNJK(
            in_feats=features.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=nb_classes,
            model=args.model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            jk_mode=args.jk_mode
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        best_val_f1 = -1
        best_epoch = -1
        best_metrics = {}

        for epoch in range(args.epochs):
            model.train()
            logits = model(features, edge_index)
            loss = loss_fn(logits[train_mask], labels_tensor[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                val_pred = pred[val_mask].cpu().numpy()
                val_true = labels_tensor[val_mask].cpu().numpy()
                val_f1 = f1_score(val_true, val_pred, average="macro")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    # Test metrics at best val epoch
                    test_pred = pred[test_mask].cpu().numpy()
                    test_true = labels_tensor[test_mask].cpu().numpy()
                    best_metrics = {
                        "test_accuracy": accuracy_score(test_true, test_pred),
                        "test_f1_macro": f1_score(test_true, test_pred, average="macro"),
                        "test_f1_micro": f1_score(test_true, test_pred, average="micro"),
                        "test_precision_macro": precision_score(test_true, test_pred, average="macro", zero_division=0),
                        "test_recall_macro": recall_score(test_true, test_pred, average="macro", zero_division=0),
                        "val_f1_macro": best_val_f1,
                        "best_epoch": best_epoch,
                        "final_loss": float(loss.item())
                    }

        print(f"Best Val F1_macro: {best_val_f1:.4f} at epoch {best_epoch}")
        print(f"Test F1_macro at best epoch: {best_metrics['test_f1_macro']:.4f}")
        all_results.append(best_metrics)

    # Aggregate results over seeds
    metrics = list(all_results[0].keys())
    print("\n=== Summary Across Seeds ===")
    for m in metrics:
        vals = [r[m] for r in all_results]
        print(f"{m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

if __name__ == "__main__":
    main()

