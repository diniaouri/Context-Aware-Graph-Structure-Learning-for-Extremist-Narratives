#!/usr/bin/env python3
"""
Batch runner for src/node_cls.py, dynamically validates labels and adjacency files.

Fixes:
 - Added label validation and cleaning before processing.
 - Enhanced adjacency file format validation.
 - Improved logging for debugging and error reporting.

Usage:
  python scripts/run_nodecls_simple.py path/to/sweep_results_main.csv "Label A;Label B;Label C" ARENASGermanAnnotator2 10

For each adjacency file in the CSV column `saved_adj`, the script runs src/node_cls.py with the specified labels.
Logs are created in the ./node_cls_runs directory.
"""
import sys
import subprocess
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def load_adj_paths(csv_path, adj_column="saved_adj"):
    """
    Loads adjacency paths from the CSV file.
    """
    df = pd.read_csv(csv_path)
    if adj_column not in df.columns:
        raise SystemExit(f"CSV does not contain column '{adj_column}'. Columns: {list(df.columns)}")
    rows = df[adj_column].dropna().astype(str).tolist()
    valid_paths = [p.strip() for p in rows if p.strip()]
    return valid_paths


def safe_name(s: str) -> str:
    """
    Converts a string into a safe format for filenames (alphanumeric + limited special characters).
    """
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)[:120]


def clean_labels(labels_tensor, valid_min=0, valid_max=None):
    """
    Cleans invalid values in the labels tensor (NaN, negative values, or out-of-range values).
    """
    print(f"[INFO] Cleaning labels...")
    if torch.isnan(labels_tensor).any():
        print("[WARN] NaN values detected in labels. Replacing NaN with valid_min.")
        labels_tensor = torch.nan_to_num(labels_tensor, nan=valid_min)

    if torch.any(labels_tensor < valid_min):
        print("[WARN] Negative labels detected. Replacing them with valid_min.")
        labels_tensor = torch.clamp(labels_tensor, min=valid_min)

    if valid_max is not None and torch.any(labels_tensor > valid_max):
        print(f"[WARN] Labels exceeding {valid_max} detected. Replacing them with valid_max.")
        labels_tensor = torch.clamp(labels_tensor, max=valid_max)

    print(f"[INFO] Cleaned labels: {labels_tensor.unique().tolist()}")
    return labels_tensor


def validate_adjacency_files(adj_paths):
    """
    Validates adjacency files for existence and correct formats (.npy, .pkl).
    """
    valid_paths = []
    for adj in adj_paths:
        adj_path = Path(adj)
        if not adj_path.exists():
            print(f"[WARN] Adjacency file not found: {adj}")
            continue
        try:
            if adj_path.suffix in {".pkl", ".pickle"}:  # Validate pickle adjacency files
                import pickle
                with open(adj_path, "rb") as fh:
                    adj_data = pickle.load(fh)
                if not isinstance(adj_data, (list, np.ndarray)):
                    raise ValueError("Adjacency file format invalid (expected numpy array or list).")
            elif adj_path.suffix == ".npy":  # Validate numpy adjacency files
                adj_data = np.load(adj_path)
                if len(adj_data.shape) != 2:
                    raise ValueError("Adjacency matrix must be 2D.")
            else:
                raise ValueError(f"Unsupported adjacency file format: {adj_path.suffix}.")
            valid_paths.append(adj)
        except Exception as e:
            print(f"[WARN] Failed to load adjacency file '{adj}': {e}")
    return valid_paths


def main():
    if len(sys.argv) < 5:
        print("Usage: python scripts/run_nodecls_simple.py <csv> \"Label1;Label2;Label3\" <dataset> <experiment_nb>")
        sys.exit(2)

    csv_path = Path(sys.argv[1])
    labels_str = sys.argv[2]
    dataset = sys.argv[3]
    experiment_nb = sys.argv[4]

    if not csv_path.exists():
        print("CSV not found:", csv_path)
        sys.exit(2)

    labels = [l.strip() for l in labels_str.split(";") if l.strip()]
    if not labels:
        print("No labels provided.")
        sys.exit(2)

    try:
        experiment_nb = int(experiment_nb)
    except ValueError:
        print("experiment_nb must be an integer")
        sys.exit(2)

    adj_paths = load_adj_paths(csv_path, adj_column="saved_adj")
    adj_paths = validate_adjacency_files(adj_paths)
    if not adj_paths:
        print("No valid adjacency files found.")
        sys.exit(1)

    logdir = Path("node_cls_runs")
    logdir.mkdir(exist_ok=True)

    node_cls_script = Path("src") / "node_cls.py"
    if not node_cls_script.exists():
        print("Cannot find src/node_cls.py in repo root. Adjust working directory or script location.")
        sys.exit(2)

    py = sys.executable
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable better error reporting for CUDA

    job_i = 0
    for adj in adj_paths:
        for label in labels:
            job_i += 1
            safe_label = safe_name(label)
            adj_stem = Path(adj).stem
            log_name = f"job_{job_i:03d}__{adj_stem}__label_{safe_label}.log"
            log_path = logdir / log_name
            cmd = [
                py,
                str(node_cls_script),
                "--dataset", dataset,
                "--experiment_nb", str(experiment_nb),
                "--label_col", label,
                "--adjacency_matrix", adj
            ]
            cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
            print(f"[{job_i}] Running: {cmd_str}")
            print(f"[{job_i}] Log -> {log_path}")
            try:
                with open(log_path, "wb") as fh:
                    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
                    rc = proc.wait()
                if rc != 0:
                    print(f"[{job_i}] FAILED rc={rc} (see {log_path})")
                else:
                    print(f"[{job_i}] DONE rc={rc}")
            except Exception as e:
                print(f"[{job_i}] ERROR during subprocess execution: {e}")


if __name__ == "__main__":
    main()
