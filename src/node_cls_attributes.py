import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle

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
    parser = argparse.ArgumentParser(description="Node classification with PyG (no DGL!)")
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--adjacency_matrix', type=str, required=True)
    parser.add_argument('--experiment_nb', type=int, default=3)
    parser.add_argument('--embeddings_file', type=str, default=None)
    parser.add_argument('--feature_cols', type=str, nargs='+', required=True)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--cpu_only', action='store_true')
    return parser.parse_args()

class GraphSAGEFinetune(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    all_runs_results = []
    per_run_avgs = []

    for run in range(args.n_runs):
        run_seed = args.seed + run
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)

        # Load adjacency matrix
        with open(args.adjacency_matrix, 'rb') as f:
            adj_matrix = pickle.load(f)
        adj = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix

        # Load dataset
        dataset = DATASETS[args.dataset](experiment_nb=args.experiment_nb, embeddings_path=args.embeddings_file)
        node_features = dataset.data
        needed_cols = args.feature_cols + [args.label_col]
        node_features = node_features.dropna(subset=needed_cols).reset_index(drop=True)

        # Label encoding
        unique_labels = node_features[args.label_col].unique()
        label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
        node_features['label'] = node_features[args.label_col].map(label_to_num)

        # Remove classes with <n_splits samples to prevent stratified split errors
        class_counts = node_features['label'].value_counts()
        keep_classes = class_counts[class_counts >= args.n_splits].index
        before_count = len(node_features)
        node_features = node_features[node_features['label'].isin(keep_classes)]
        node_features = node_features.reset_index(drop=False)  # keep original index as column
        after_count = len(node_features)
        if after_count < before_count:
            print(f"Warning: {before_count - after_count} samples dropped because their class had fewer than {args.n_splits} instances (n_splits={args.n_splits}).")

        labels = node_features['label'].values
        nb_classes = len(np.unique(labels))

        # Filter adjacency matrix to only the remaining nodes
        remaining_idx = node_features['index'].values
        adj = adj[np.ix_(remaining_idx, remaining_idx)]

        # Compose feature matrix
        features_list = []
        if args.embeddings_file is not None:
            sublime_embs_all = np.load(args.embeddings_file)
            sublime_embs = sublime_embs_all[node_features['index'].values]
            assert sublime_embs.shape[0] == len(node_features), "SUBLIME embeddings count does not match nodes"
            features_list.append(sublime_embs)
        for col in args.feature_cols:
            if node_features[col].dtype == object or node_features[col].dtype.name == "category":
                onehots = pd.get_dummies(node_features[col], prefix=col)
                features_list.append(onehots.values)
            else:
                arr = node_features[[col]].values.astype(np.float32)
                features_list.append(arr)
        features = np.concatenate(features_list, axis=1)
        features = torch.tensor(features, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        assert adj.shape[0] == len(node_features)
        assert features.shape[0] == len(node_features)

        node_features = node_features.drop(columns=['index']).reset_index(drop=True)

        # PyG edge_index from adjacency
        src, dst = np.nonzero(adj)
        edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=run_seed)
        run_results = []

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros_like(labels), labels)):
            # 60/20/20 split: train/val/test
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=0.25,  # 25% of 80% = 20% of total
                stratify=labels[trainval_idx],
                random_state=run_seed + fold
            )
            print(f"\n=== Run {run+1}/{args.n_runs}, Fold {fold + 1}/{args.n_splits} ===")

            model = GraphSAGEFinetune(features.shape[1], args.hidden_dim, nb_classes, args.num_layers, args.dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            loss_fn = nn.CrossEntropyLoss()
            train_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
            val_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
            test_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            best_val_f1 = -1
            best_epoch = -1
            best_metrics = {}

            for epoch in range(args.epochs):
                model.train()
                out = model(features, edge_index)
                loss = loss_fn(out[train_mask], labels_tensor[train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    logits = model(features, edge_index)
                    pred_class = logits.argmax(dim=1)
                    val_pred = pred_class[val_mask].cpu().numpy()
                    val_true = labels_tensor[val_mask].cpu().numpy()
                    val_f1 = f1_score(val_true, val_pred, average="macro")
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        # Save test metrics at this epoch
                        test_pred = pred_class[test_mask].cpu().numpy()
                        test_true = labels_tensor[test_mask].cpu().numpy()
                        acc = accuracy_score(test_true, test_pred)
                        f1_macro = f1_score(test_true, test_pred, average="macro")
                        f1_micro = f1_score(test_true, test_pred, average="micro")
                        precision_macro = precision_score(test_true, test_pred, average="macro", zero_division=0)
                        recall_macro = recall_score(test_true, test_pred, average="macro", zero_division=0)
                        precision_micro = precision_score(test_true, test_pred, average="micro", zero_division=0)
                        recall_micro = recall_score(test_true, test_pred, average="micro", zero_division=0)
                        best_metrics = {
                            "test_accuracy": acc,
                            "test_f1_macro": f1_macro,
                            "test_f1_micro": f1_micro,
                            "test_precision_macro": precision_macro,
                            "test_recall_macro": recall_macro,
                            "test_precision_micro": precision_micro,
                            "test_recall_micro": recall_micro,
                            "val_f1_macro": best_val_f1,
                            "best_epoch": best_epoch,
                        }
            print(f"Best Val F1_macro: {best_val_f1:.4f} at epoch {best_epoch}")
            print(f"Test F1_macro at best epoch: {best_metrics['test_f1_macro']:.4f}")
            run_results.append(best_metrics)
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

    print(f"\n=== Average of the average of the {args.n_runs} runs ===")
    for metric in metrics:
        per_run_metric_avgs = [run_avg[metric] for run_avg in per_run_avgs]
        avg = np.mean(per_run_metric_avgs)
        std = np.std(per_run_metric_avgs)
        print(f"{metric}: {avg:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
