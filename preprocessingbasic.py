import pandas as pd
import numpy as np
import re
import emoji
from sentence_transformers import SentenceTransformer
import torch
import os


class GraphDataset():

    def __init__(self):
        # Open dataset (stored as excel sheet) and remove useless rows / columns
        self.data = pd.read_excel("2024_08_SCHEMA_A1.xlsx",
                                  header=4, usecols="B, AH:BA")
        # Remove rows that don't have text or in group or out group
        self.data = self.data.dropna(subset=["Text", "In-Group", "Out-group"])
        self.data = self.data.reset_index(drop=True)
        self.data["Text"] = self.data["Text"].apply(self.clean_up_text)
        self.data.to_csv("ty.csv")
        # Embedding options : jinaai/jina-embeddings-v3
        #                     dangvantuan/french-document-embedding
        #                     almanach/camembertav2-base
        #                     sentence-transformers/all-MiniLM-L12-v2
        self.embeddings = self.calc_embeddings(
            "dangvantuan/french-document-embedding")
        np.save("SCHEMA_A1_embeddings_exp2.npy", self.embeddings)

        self.embeddings = torch.from_numpy(self.embeddings)
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.labels = []
        self.adjacency_matrix = []

    def calc_embeddings(self, model_name):
        # Initialize the model
        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Create embeddings
        embeddings = model.encode(
            self.data["Text"].tolist(),
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def clean_up_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Convert emojis to text
        text = emoji.demojize(text)

        # Remove @mentions
        text = re.sub(r'@\w+', '', text)

        # Remove hashtag symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove RT (retweet) indicator
        text = re.sub(r'^RT[\s]+', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing spaces
        text = text.strip()

        # Convert to lowercase
        text = text.lower()

        return text

    def get_dataset(self):
        return self.embeddings, self.embeddings.shape[1], self.labels, len(self.labels), self.train_mask, self.val_mask, self.test_mask, self.adjacency_matrix


if __name__ == "__main__":
    first_dataset = GraphDataset()
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj = first_dataset.get_dataset()
    print(first_dataset.data)
