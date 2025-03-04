"""
This module provides classes and functions to preprocess datasets,
calculate text embeddings using SentenceTransformer, and prepare the data
for downstream tasks.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import emoji
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class PreprocessedDataset(ABC):
    """
    Abstract base class for preprocessing datasets and computing embeddings.
    """

    def __init__(self, experiment_nb: int) -> None:
        """
        Initialize the dataset by cleaning it up and computing embeddings.

        Args:
            dataset_file_name (str): The name of the dataset file.
            experiment_nb (int): The experiment number identifier.
        """
        self.experiment: int = experiment_nb

        self.data: pd.DataFrame = self.clean_up_dataset()
        # Embedding options : jinaai/jina-embeddings-v3
        #                     dangvantuan/french-document-embedding
        #                     almanach/camembertav2-base
        #                     sentence-transformers/all-MiniLM-L12-v2
        # Just so I don't have to remember
        self.calc_embeddings_if_not_already(
            "dangvantuan/french-document-embedding")
        print("Embeddings are loaded")

        self.embeddings = torch.from_numpy(self.embeddings)
        self.train_mask: List[Any] = []
        self.val_mask: List[Any] = []
        self.test_mask: List[Any] = []
        self.labels: List[Any] = []
        self.adjacency_matrix: List[Any] = []

    def save_embeddings(self) -> None:
        """
        Save the calculated embeddings to a .npy file.
        """
        os.makedirs("./embeddings", exist_ok=True)
        np.save(
            f"./embeddings/{self.dataset_name}_embeddings_exp{self.experiment}.npy",
            self.embeddings
        )

    def calc_embeddings_if_not_already(self, model_name):
        """Only calculate embeddings if not already calculated

        Args:
            model_name (str): The name of the SentenceTransformer model to use.

        Returns:
            np.ndarray: The calculated embeddings as a NumPy array.
        """
        print()
        if not os.path.exists(f"./embeddings/{self.dataset_name}_embeddings_exp{self.experiment}.npy"):
            self.embeddings = self.calc_embeddings(model_name)
            self.save_embeddings()
        else:
            self.embeddings = np.load(
                f"./embeddings/{self.dataset_name}_embeddings_exp{self.experiment}.npy")

    def calc_embeddings(self, model_name: str) -> np.ndarray:
        """
            Calculate embeddings for the cleaned dataset using the specified model.

            Args:
                model_name (str): The name of the SentenceTransformer model to use.

            Returns:
                np.ndarray: The calculated embeddings as a NumPy array.
            """
        # Initialize the model with remote code trust enabled.
        model = SentenceTransformer(model_name, trust_remote_code=True)

        # Create embeddings from the 'Text' column.
        embeddings: np.ndarray = model.encode(
            self.data["Text"].tolist(),
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def get_dataset(self) -> Tuple[torch.Tensor, int, List[Any], int, List[Any], List[Any], List[Any], List[Any]]:
        """
        Retrieve the processed dataset in the format expected by the SUBLIME code.

        Note:
            Since we are not using the node classification downstream task, the labels and masks remain empty.
            The adjacency matrix is empty as we are only using the structure inference context.

        Returns:
            Tuple containing:
                - embeddings (torch.Tensor): The embeddings tensor.
                - n_features (int): The number of features in each embedding.
                - labels (List[Any]): Labels (empty list).
                - n_classes (int): Number of classes (derived from labels length).
                - train_mask (List[Any]): Training mask (empty list).
                - val_mask (List[Any]): Validation mask (empty list).
                - test_mask (List[Any]): Test mask (empty list).
                - adjacency_matrix (List[Any]): Adjacency matrix (empty list).
        """
        return (
            self.embeddings,
            self.embeddings.shape[1],
            self.labels,
            len(self.labels),
            self.train_mask,
            self.val_mask,
            self.test_mask,
            self.adjacency_matrix
        )

    @abstractmethod
    def clean_up_dataset(self) -> pd.DataFrame:
        """
        Perform dataset-specific cleaning and return a cleaned DataFrame.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        pass

    def clean_up_text(self, text: str) -> str:
        """
        Clean up the input text by performing several operations:
        - Remove URLs
        - Convert emojis to text
        - Remove @mentions
        - Remove hashtag symbols but keep the text
        - Remove RT (retweet) indicators
        - Remove multiple spaces and trim
        - Convert to lowercase

        Args:
            text (str): The text string to clean.

        Returns:
            str: The cleaned text string.
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Convert emojis to their textual representation
        text = emoji.demojize(text)

        # Remove @mention
        text = re.sub(r'@\w+', '', text)

        # Remove hashtag symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove RT (retweet) indicator
        text = re.sub(r'^RT[\s]+', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing spaces and convert to lowercase
        text = text.strip().lower()

        return text


def remove_file_extension(filename: str) -> str:
    """
    Remove the file extension from a filename.

    Args:
        filename (str): The filename from which to remove the extension.

    Returns:
        str: The filename without its extension.
    """
    return os.path.splitext(filename)[0]


class FullFrenchTweetDataset(PreprocessedDataset):
    """
    Concrete implementation of PreprocessedDataset for French tweet data.
    """

    def __init__(self, experiment_nb: int) -> None:
        """
        Initialize the FullFrenchTweetDataset.

        Args:
            experiment_nb (int): The experiment number.
        """
        self.dataset_name: str = "All_french_tweet_data"
        super().__init__(experiment_nb)
        # self.calc_mix_of_features_embeddings_if_not_already()

    def calc_mix_of_features_embeddings_if_not_already(self):
        """Only calculate embeddings if not already calculated

        Args:
            model_name (str): The name of the SentenceTransformer model to use.

        Returns:
            np.ndarray: The calculated embeddings as a NumPy array.
        """
        if not os.path.exists(f"./embeddings/{self.dataset_name}_mixed_features_embeddings_exp{self.experiment}.npy"):
            self.mixed_features_embeddings = self.calc_mix_of_features_embeddings()
        else:
            self.mixed_features_embeddings = np.load(
                f"./embeddings/{self.dataset_name}_mixed_features_embeddings_exp{self.experiment}.npy")

    def calc_mix_of_features_embeddings(self):
        """
        Calculate embeddings for the cleaned dataset using the specified model.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.

        Returns:
            np.ndarray: The calculated embeddings as a NumPy array.
        """
        # Initialize the model with remote code trust enabled.
        model = SentenceTransformer(
            "dangvantuan/french-document-embedding", trust_remote_code=True)

        # Create embeddings from the 'Text' column.
        embeddings: np.ndarray = model.encode(
            self.data["Mixed features"].tolist(),
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        np.save(
            f"./embeddings/{self.dataset_name}_mixed_features_embeddings_exp{self.experiment}.npy",
            embeddings
        )

    def clean_up_dataset(self) -> pd.DataFrame:
        """
        Clean up the French tweet dataset:
        - Load the dataset from a CSV file.
        - Select specific columns.
        - Rename columns.
        - Remove retweets.
        - Randomly sample a subset of tweets.
        - Clean the text of each tweet.

        Returns:
            pd.DataFrame: The cleaned dataset as a DataFrame.
        """
        # Load dataset from CSV
        df: pd.DataFrame = pd.read_csv("datasets/All_french_tweet_data.csv")
        df = df[["Tweet, Text", "User, name", "User, Description"]]

        df = df.rename(columns={"Tweet, Text": "Text", "User, name": "User"})
        df["Mixed features"] = df["Text"] + \
            df["User"] + df["User, Description"]
        # Remove retweets (posts that start with "RT")
        df = df[~df["Text"].str.startswith("RT", na=False)].copy()

        # Randomly sample n < 30000 tweets from approximately 30000 tweets to make experiments faster.
        df = df.sample(n=10000, random_state=42)
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text, args=(True,))
        df["Mixed features"] = df["Mixed features"].apply(
            self.clean_up_text, args=(False,))
        df.to_csv("ty.csv", index=False)
        return df

    def clean_up_text(self, text: str, remove_mentions=True) -> str:

        if not isinstance(text, str):
            return "Erreur"
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Convert emojis to their textual representation
        text = emoji.demojize(text)

        if remove_mentions:
            # Remove @mention
            text = re.sub(r'@\w+', '', text)
        else:
            # Remove @ symbol but keep mentions text
            text = re.sub(r'@(\w+)', '', text)

        # Remove hashtag symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)

        # Remove RT (retweet) indicator
        text = re.sub(r'^RT[\s]+', '', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing spaces and convert to lowercase
        text = text.strip().lower()

        return text


class SchemaA1Dataset(PreprocessedDataset):
    """
    Concrete implementation of PreprocessedDataset for SCHEMA A1.
    """

    def __init__(self, experiment_nb: int) -> None:
        """
        Initialize the class.

        Args:
            dataset_file_name (str): The name of the dataset file.
            experiment_nb (int): The experiment number.
        """
        self.dataset_name: str = "2024_08_SCHEMA_A1"
        super().__init__(experiment_nb)

    def clean_up_dataset(self) -> pd.DataFrame:
        """
        Clean up the dataset:
        - Load the dataset from a xlsx file.
        - Select specific columns.
        - Remove rows that have empty values in specific columns
        - Clean the text of each tweet.

        Returns:
            pd.DataFrame: The cleaned dataset as a DataFrame.
        """
        df = pd.read_excel("./datasets/2024_08_SCHEMA_A1.xlsx",
                           header=4, usecols="B, AH:BA")
        # Remove rows that don't have text or in group or out group
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty.csv")
        return df


class OldSchemaA1Dataset(PreprocessedDataset):
    """
    Concrete implementation of PreprocessedDataset for SCHEMA A1.
    """

    def __init__(self, experiment_nb: int) -> None:
        """
        Initialize the class.

        Args:
            dataset_file_name (str): The name of the dataset file.
            experiment_nb (int): The experiment number.
        """
        self.dataset_name: str = "2024_08_SCHEMA_A1"
        super().__init__(experiment_nb)

    def clean_up_dataset(self) -> pd.DataFrame:
        """
        Clean up the dataset:
        - Load the dataset from a xlsx file.
        - Select specific columns.
        - Remove rows that have empty values in specific columns
        - Clean the text of each tweet.

        Returns:
            pd.DataFrame: The cleaned dataset as a DataFrame.
        """
        df = pd.read_excel("./datasets/2024_08_SCHEMA_A1.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text"])
        df = df.reset_index()
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df


class SelectedDataset(PreprocessedDataset):

    def __init__(self, experiment_nb):
        self.dataset_name = "300*3Tweets_ARENAS"
        super().__init__(experiment_nb)

    def clean_up_dataset(self):
        df1 = pd.read_csv("datasets/300Tweets_ARENAS_Gender.csv")
        df2 = pd.read_csv("datasets/300Tweets_ARENAS_Nation.csv")
        df3 = pd.read_csv("datasets/300Tweets_ARENAS_Science.csv")
        df = pd.concat([df1, df2, df3], axis=0)
        df = df.reset_index(drop=True)
        df = df.rename(columns={"Document": "Text"})
        return df


if __name__ == "__main__":
    first_dataset = SelectedDataset(5)
    features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj = first_dataset.get_dataset()
    print(first_dataset.data)
