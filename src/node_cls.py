import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GCNConv, BatchNorm, JumpingKnowledge
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
    parser = argparse.ArgumentParser(
        description="Unified node classification with PyG (JK + BatchNorm) and optional attribute features."
    )
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--label_col', type=str, required=True)
    parser.add_argument('--adjacency_matrix', type=str, required=True)
    parser.add_argument('--experiment_nb', type=int, default=3)

    # Features
    parser.add_argument('--embeddings_file', type=str, default=None,
                        help="Path to .npy embeddings. Optional if you only use --feature_cols.")
    parser.add_argument('--feature_cols', type=str, nargs='*', default=None,
                        help='Optional list of attribute columns to include as features (e.g., "In-Group" "Out-group").')

    # Model
    parser.add_argument('--model', type=str, default='GraphSAGE', choices=['GCN', 'GraphSAGE'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--jk_mode', type=str, default='cat', choices=['cat', 'max', 'lstm', 'sum'])

    # Training
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--cpu_only', action='store_true')
    return parser.parse_args()

class GNNWithBNJK(nn.Module):
    """
    GCN/GraphSAGE with BatchNorm per layer and Jumping Knowledge aggregation.
    Final linear head maps JK output to num_classes.
    """
    def __init__(self, in_feats, hidden_dim, num_classes, model='GraphSAGE',
                 num_layers=3, dropout=0.2, jk_mode='cat'):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.model = model
        self.num_layers = num_layers
        self.dropout = dropout

        conv_class = GCNConv if model == 'GCN' else SAGEConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(conv_class(in_feats, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(conv_class(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        self.jk = JumpingKnowledge(jk_mode)
        out_feats = hidden_dim * num_layers if jk_mode == 'cat' else hidden_dim
        self.lin = nn.Linear(out_feats, num_classes)

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jk(xs)
        x = self.lin(x)  # logits
        return x

def build_features(node_df, remaining_idx, embeddings_file, feature_cols):
    """
    Build feature matrix by concatenating:
      - embeddings_file (optional)
      - one-hot or numeric columns from feature_cols (optional)
    Returns np.ndarray [N_nodes, D]
    """
    feats = []

    if embeddings_file is not None:
        all_embs = np.load(embeddings_file)
        embs = all_embs[remaining_idx]
        feats.append(embs)

    if feature_cols:
        for col in feature_cols:
            col_data = node_df.loc[:, col]
            if col_data.dtype == object or col_data.dtype.name == "category":
                onehots = pd.get_dummies(col_data, prefix=col)
                feats.append(onehots.values.astype(np.float32))
            else:
                arr = node_df[[col]].values.astype(np.float32)
                feats.append(arr)

    if not feats:
        raise ValueError("No features provided. Supply --embeddings_file and/or --feature_cols.")
    return np.concatenate(feats, axis=1).astype(np.float32)

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

        # Load adjacency matrix (pickle with numpy array or torch tensor)
        with open(args.adjacency_matrix, 'rb') as f:
            adj_matrix = pickle.load(f)
        adj = adj_matrix.cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix

        # Load dataset
        dataset = DATASETS[args.dataset](experiment_nb=args.experiment_nb, embeddings_path=args.embeddings_file)
        df = dataset.data

        # Drop rows missing required columns (label + any requested feature cols)
        required_cols = [args.label_col] + (args.feature_cols or [])
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        # Encode labels
        unique_labels = df[args.label_col].unique()
        label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
        df['label'] = df[args.label_col].map(label_to_num)

        # Remove classes with < n_splits samples
        class_counts = df['label'].value_counts()
        keep_classes = class_counts[class_counts >= args.n_splits].index
        before = len(df)
        df = df[df['label'].isin(keep_classes)].reset_index(drop=False)  # keep original index
        after = len(df)
        if after < before:
            print(f"Warning: {before - after} samples dropped (class count < n_splits={args.n_splits}).")

        labels = df['label'].values
        nb_classes = len(np.unique(labels))

        # Filter adjacency to remaining nodes
        remaining_idx = df['index'].values
        adj = adj[np.ix_(remaining_idx, remaining_idx)]

        # Build features (embeddings optional + attributes optional)
        features_np = build_features(
            node_df=df.set_index(np.arange(len(df))),  # clean integer index for get_dummies alignment
            remaining_idx=np.arange(len(df)),          # features built after filtering: use local indices
            embeddings_file=args.embeddings_file,
            feature_cols=args.feature_cols
        )

        # To tensors
        features = torch.tensor(features_np, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        # Edge index from dense adjacency
        src, dst = np.nonzero(adj)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=run_seed)
        run_results = []

        for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros_like(labels), labels)):
            # 60/20/20 split => from the 80% trainval, keep 75% train and 25% val
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=0.25, stratify=labels[trainval_idx], random_state=run_seed + fold
            )

            print(f"\n=== Run {run+1}/{args.n_runs}, Fold {fold + 1}/{args.n_splits} ===")
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

            # Masks
            N = features.shape[0]
            train_mask = torch.zeros(N, dtype=torch.bool, device=device); train_mask[train_idx] = True
            val_mask   = torch.zeros(N, dtype=torch.bool, device=device); val_mask[val_idx]   = True
            test_mask  = torch.zeros(N, dtype=torch.bool, device=device); test_mask[test_idx]  = True

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
                    logits = model(features, edge_index)
                    pred = logits.argmax(dim=1)
                    val_pred = pred[val_mask].cpu().numpy()
                    val_true = labels_tensor[val_mask].cpu().numpy()
                    val_f1 = f1_score(val_true, val_pred, average="macro")
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch
                        # Evaluate test at best val epoch
                        test_pred = pred[test_mask].cpu().numpy()
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
        run_avg = {m: np.mean([fold_res[m] for fold_res in run_results]) for m in metrics}
        per_run_avgs.append(run_avg)
        print(f"\n=== Average for Run {run+1}/{args.n_runs} ===")
        for m in metrics:
            vals = [fold_res[m] for fold_res in run_results]
            print(f"  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Aggregate across all runs and folds
    metrics = list(all_runs_results[0][0].keys())
    all_metrics = {m: [] for m in metrics}
    for run_results in all_runs_results:
        for fold_res in run_results:
            for m in metrics:
                all_metrics[m].append(fold_res[m])

    print("\n=== Overall Results Across All Runs and Folds ===")
    for m in metrics:
        avg = np.mean(all_metrics[m]); std = np.std(all_metrics[m])
        print(f"{m}: {avg:.4f} ± {std:.4f}")

    print(f"\n=== Average of the average of the {args.n_runs} runs ===")
    for m in metrics:
        per_run_metric_avgs = [run_avg[m] for run_avg in per_run_avgs]
        avg = np.mean(per_run_metric_avgs); std = np.std(per_run_metric_avgs)
        print(f"{m}: {avg:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()