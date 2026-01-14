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
    def __init__(self, df, tokenizer, text_col, label_col, feature_cols=None, max_length=128):
        self.texts = df[text_col].fillna("").astype(str).tolist()
        self.labels = df[label_col].tolist()
        self.features = df[feature_cols].fillna(0).values if feature_cols else None
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

        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float32)

        return item


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Text Classification with seeds-based splitting by multiple targets")
    parser.add_argument('--data_path', type=str, required=True, help='Path to your CSV or XLSX file')
    parser.add_argument('--text_col', type=str, default='Text', help='Text column name')
    parser.add_argument('--targets', type=str, nargs='+', required=True, help='Target columns for classification (e.g., "Topic", "Intolerance")')
    parser.add_argument('--feature_cols', type=str, nargs='+', default=None, help='Feature column names (e.g., In-Group, Out-group)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    parser.add_argument('--seeds', type=int, nargs='*', default=[0, 21, 42, 84, 123], help='Random seeds for repeated runs')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.6, 0.2, 0.2], help='Train, validation, test split ratios')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    return parser.parse_args()


def prepare_data(data_path, text_col, label_col, feature_cols, tokenizer, max_length):
    ext = os.path.splitext(data_path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(data_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(data_path, header=4)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    df.columns = df.columns.str.strip()
    print("Columns found:", list(df.columns))

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found! Available columns: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Target column '{label_col}' not found! Available columns: {list(df.columns)}")
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found! Available columns: {list(df.columns)}")

    df = df.dropna(subset=[text_col, label_col])  # Drop rows with missing text or labels

    # Encode labels
    unique_labels = [l for l in df[label_col].unique() if pd.notnull(l)]
    label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
    df[label_col] = df[label_col].map(label_map)

    # Preprocess feature columns: Factorize categorical data or normalize numerical data
    if feature_cols:
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = pd.factorize(df[col])[0]  # Factorize categorical data to integers
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(np.float32)  # Ensure numerical columns are float32
            else:
                raise ValueError(f"Unsupported feature column type: {col}")

    dataset = TextClassificationDataset(df, tokenizer, text_col, label_col, feature_cols, max_length)
    return dataset, len(label_map)


def split_dataset(dataset, ratios, seed):
    lengths = [int(r * len(dataset)) for r in ratios]
    lengths[-1] = len(dataset) - sum(lengths[:-1])  # Ensure all samples are included
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))


def train_and_eval(model, train_loader, val_loader, test_loader, device, epochs, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items() if k != 'features'}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

    model.eval()

    def evaluate(loader):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch['labels'].cpu().numpy()
                batch = {k: v.to(device) for k, v in batch.items() if k != 'features'}
                outputs = model(**batch)
                logits = outputs.logits.cpu().numpy()
                preds = np.argmax(logits, axis=1)
                all_preds.extend(preds)
                all_labels.extend(labels)

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "f1_micro": f1_score(all_labels, all_preds, average="micro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "precision_micro": precision_score(all_labels, all_preds, average="micro", zero_division=0),
            "recall_micro": recall_score(all_labels, all_preds, average="micro", zero_division=0),
        }

    return {
        "validation": evaluate(val_loader),
        "test": evaluate(test_loader),
    }


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU device: {args.gpu}")
    else:
        print("CUDA is not available; using CPU.")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for target in args.targets:
        print(f"\n=== Running experiments for target '{target}' ===")
        dataset, n_labels = prepare_data(args.data_path, args.text_col, target, args.feature_cols, tokenizer, args.max_length)

        all_results = []
        for seed in args.seeds:
            print(f"\n=== Run with Seed {seed} ===")
            train_data, val_data, test_data = split_dataset(dataset, args.split_ratios, seed)
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size)
            test_loader = DataLoader(test_data, batch_size=args.batch_size)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=n_labels
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=args.lr)

            results = train_and_eval(model, train_loader, val_loader, test_loader, device, args.epochs, optimizer)
            all_results.append(results)

        print(f"\n=== Overall Results for Target '{target}' ===")
        for metric in all_results[0]['validation'].keys():
            val_scores = [result['validation'][metric] for result in all_results]
            test_scores = [result['test'][metric] for result in all_results]
            print(f"{metric} (Validation): {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
            print(f"{metric} (Test): {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")


if __name__ == "__main__":
    main()
