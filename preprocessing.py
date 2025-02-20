import pandas as pd
import numpy as np
import re
import emoji
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
import torch
import os


class GraphDataset():

    def __init__(self, adjacency_matrix_type="None", label_column="In-Group"):
        # Open dataset (stored as excel sheet) and remove useless rows / columns
        self.data = pd.read_excel("./datasets/2024_08_SCHEMA_A1.xlsx",
                                  header=4, usecols="B, AH:BA")
        self.data = self.data.dropna(subset=["Text"])
        self.label_column = label_column
        print(self.data[label_column].unique())
        print(len(self.data[label_column].unique()))
        print(len(self.data))
        # We are removing certain out group classes because those have too little labels in the dataset to perform a train test split
        # values_to_keep = ["Scientists/Academics",
        #                   "Media/Journalists", "Feminists", "Politicians/Government", "The People", "LGBTIQ+", "Antivax/Science Skeptics"]
        values_to_keep = ['LGBTIQ+', 'The People', 'Politicians/Government',
                          'Feminists', 'European Actors', 'Media/Journalists',
                          'The Elite/Establishment', 'Right',
                          'Nation/Own Country',
                          'Scientists/Academics',
                          'Immigrants/Asylum Seekers',
                          'Antivax/Science Skeptics']
        values_to_keep = ["Political Party",
                          "Right", "Feminists", "The People"]
        # self.data = self.data[self.data[label_column].isin(values_to_keep)]
        self.data = self.data.reset_index()
        print(len(self.data))
        self.data["Text"] = self.data["Text"].apply(self.clean_up_text)
        # Calculate or load text embeddings
        if not os.path.exists("2024_08_SCHEMA_A1_embeddings.npy"):
            print("Calculating embeddings\n")

            # Embedding options : jinaai/jina-embeddings-v3
            #                     dangvantuan/french-document-embedding
            #                     almanach/camembertav2-base
            #                     sentence-transformers/all-MiniLM-L12-v2
            self.embeddings = self.calc_embeddings(
                "dangvantuan/french-document-embedding")
            np.save("2024_08_SCHEMA_A1_embeddings.npy", self.embeddings)
        else:
            print("Loading Embeddings\n")
            self.embeddings = np.load("2024_08_SCHEMA_A1_embeddings.npy")
            # If nb rows in embeddings don't match the nb rows in data, recalculate embeddings
            if self.embeddings.shape[0] != len(self.data):
                print("Calculating embeddings\n")
                self.embeddings = self.calc_embeddings(
                    "dangvantuan/french-document-embedding")
                np.save("2024_08_SCHEMA_A1_embeddings.npy", self.embeddings)
        self.embeddings = torch.from_numpy(self.embeddings)

        if adjacency_matrix_type == "topic":
            self.adjacency_matrix = self.create_adj_matrix_topic()
        elif adjacency_matrix_type == "knn":
            self.adjacency_matrix = kneighbors_graph(
                self.embeddings, n_neighbors=10, metric="cosine")
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        # self.train_mask, self.val_mask, self.test_mask = self.create_masks()
        codes, uniques = pd.factorize(self.data[self.label_column])
        print(uniques)
        self.labels = torch.tensor(codes)

    def create_adj_matrix_topic(self):
        # Extract the 'Topic' column as a NumPy array
        topics = self.data['Topic'].values

        # Create a symmetric matrix using broadcasting
        matrix = (topics[:, None] == topics).astype(int)
        np.fill_diagonal(matrix, 0)
        return matrix

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

    def create_masks(self, train_size=0.2, val_size=0.2, test_size=0.6, random_state=42):
        # Initialize masks
        train_mask = np.zeros(len(self.data), dtype=bool)
        val_mask = np.zeros(len(self.data), dtype=bool)
        test_mask = np.zeros(len(self.data), dtype=bool)

        # Split for each class to maintain balance
        for label in self.data[self.label_column].unique():
            class_indices = self.data[self.data[self.label_column]
                                      == label].index
            print(f"{label} has {len(class_indices)} samples")
            train_idx, temp_idx = train_test_split(
                class_indices,
                train_size=train_size,
                random_state=random_state,
                shuffle=True
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                train_size=val_size/(test_size + val_size),
                random_state=random_state,
                shuffle=True
            )

            # Set masks
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        return torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)

    def get_dataset(self):
        return self.embeddings, self.embeddings.shape[1], self.labels, len(self.labels.unique()), self.train_mask, self.val_mask, self.test_mask, self.adjacency_matrix


def class_distribution(labels, mask):
    """
    Calculate ratio of each class in the masked portion of labels

    Args:
        labels: tensor of labels (N,)
        mask: boolean tensor (N,)

    Returns:
        Tensor of ratios for each class
    """
    masked_labels = labels[mask]
    num_classes = labels.max() + 1

    # Count occurrences and divide by total masked samples
    class_counts = torch.bincount(masked_labels, minlength=num_classes)
    ratios = class_counts.float() / len(masked_labels)

    return ratios


if __name__ == "__main__":
    first_dataset = GraphDataset(adjacency_matrix_type="knn")
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj = first_dataset.get_dataset()
    # print("---------------------Feature Matrix-------------------------------")
    # print(features)
    # print(features.shape)
    # print(f"Number of features : {nfeats}")
    # print("------------------------------Labels----------------------------")
    # print(labels)
    # print(labels.shape)
    # print(labels.unique())
    # print(f"Number of classes : {nclasses}")
    # print("---------------------------Masks----------------------------------")
    # print("Training Mask :\n")
    # print(train_mask)
    # print(train_mask.shape)
    # unique, counts = train_mask.unique(return_counts=True)
    # occurrences = dict(zip(unique.tolist(), counts.tolist()))
    # print(occurrences)
    # print("Val Mask :\n")
    # print(val_mask)
    # print(val_mask.shape)
    # unique, counts = val_mask.unique(return_counts=True)
    # occurrences = dict(zip(unique.tolist(), counts.tolist()))
    # print(occurrences)
    # print("Test Mask :\n")
    # print(test_mask)
    # print(test_mask.shape)
    # unique, counts = test_mask.unique(return_counts=True)
    # occurrences = dict(zip(unique.tolist(), counts.tolist()))
    # print(occurrences)

    print(class_distribution(labels, train_mask))
