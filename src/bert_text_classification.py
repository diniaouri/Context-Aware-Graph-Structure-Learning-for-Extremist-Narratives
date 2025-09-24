import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

class TextClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, text_col, label_col, max_length=128):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def parse_args():
    parser = argparse.ArgumentParser(description="Simple BERT Text Classification (80/20 split, 10 runs)")
    parser.add_argument('--data_path', type=str, required=True, help='Path to your CSV or XLSX')
    parser.add_argument('--text_col', type=str, default='Text', help='Text column name')
    parser.add_argument('--label_col', type=str, default='Out-group', help='Label column name')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    parser.add_argument('--runs', type=int, default=10, help='How many train/test runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def prepare_data(data_path, text_col, label_col, tokenizer, max_length):
    ext = os.path.splitext(data_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path, header=4)  # Use row 5 (index 4) as header
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    df.columns = df.columns.str.strip()
    print("Columns found:", list(df.columns))

    # Sanity check: Ensure text_col and label_col exist
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found! Available columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found! Available columns: {list(df.columns)}")

    # Drop rows where label is missing
    df = df.dropna(subset=[label_col])

    # Make label map only from non-null, non-nan
    unique_labels = [l for l in df[label_col].unique() if pd.notnull(l)]
    label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
    df[label_col] = df[label_col].map(label_map)
    dataset = TextClassificationDataset(df, tokenizer, text_col, label_col, max_length)
    return dataset, len(label_map)

def run_once(dataset, n_labels, args, run_idx):
    torch.manual_seed(args.seed + run_idx)
    np.random.seed(args.seed + run_idx)
    # 80/20 split
    train_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(args.seed + run_idx))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=n_labels
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Run {run_idx+1} Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch['labels'].cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0),
    }
    return metrics

def main():
    args = parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset, n_labels = prepare_data(args.data_path, args.text_col, args.label_col, tokenizer, args.max_length)

    all_results = []
    for run_idx in range(args.runs):
        print(f"\n=== Run {run_idx+1}/{args.runs} ===")
        run_result = run_once(dataset, n_labels, args, run_idx)
        for metric, val in run_result.items():
            print(f"{metric}: {val:.4f}")
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