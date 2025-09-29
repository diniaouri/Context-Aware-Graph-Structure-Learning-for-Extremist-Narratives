import os
import re
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional

import emoji
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

class PreprocessedDataset(ABC):
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.experiment: int = experiment_nb
        self.embeddings_path: Optional[str] = embeddings_path

        # self.data is set by subclasses BEFORE this constructor is called!
        if not skip_embeddings:
            self.calc_embeddings_if_not_already("dangvantuan/french-document-embedding")
            print("Embeddings are loaded")
            self.embeddings = torch.from_numpy(self.embeddings)
        else:
            self.embeddings = None

    def _get_embeddings_path(self) -> str:
        if self.embeddings_path is not None:
            return self.embeddings_path
        else:
            return f"./embeddings/{self.dataset_name}_embeddings_exp{self.experiment}.npy"

    def save_embeddings(self) -> None:
        path = self._get_embeddings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.embeddings)

    def calc_embeddings_if_not_already(self, model_name):
        path = self._get_embeddings_path()
        print(f"Using embeddings path: {path}")
        if not os.path.exists(path):
            self.embeddings = self.calc_embeddings(model_name)
            self.save_embeddings()
        else:
            self.embeddings = np.load(path)

    def calc_embeddings(self, model_name: str) -> np.ndarray:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        if "text" in self.data.columns and getattr(self, 'dataset_name', None) == "Toxigen":
            sentences = self.data["text"].tolist()
        else:
            if "Text" in self.data.columns:
                sentences = self.data["Text"].tolist()
            elif "text" in self.data.columns:
                sentences = self.data["text"].tolist()
            else:
                raise ValueError("No valid text column found for embeddings.")
        embeddings: np.ndarray = model.encode(
            sentences,
            batch_size=1,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def get_dataset(self) -> Tuple[torch.Tensor, int, List[Any], int, List[Any], List[Any], List[Any], List[Any]]:
        return (
            self.embeddings,
            self.embeddings.shape[1] if self.embeddings is not None else 0,
            getattr(self, 'labels', []),
            len(getattr(self, 'labels', [])),
            getattr(self, 'train_mask', []),
            getattr(self, 'val_mask', []),
            getattr(self, 'test_mask', []),
            getattr(self, 'adjacency_matrix', [])
        )

    @abstractmethod
    def clean_up_dataset(self) -> pd.DataFrame:
        pass

    def clean_up_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = emoji.demojize(text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text

def remove_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[0]

# --- Datasets ---

class FullFrenchTweetDataset(PreprocessedDataset):
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "All_french_tweet_data"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv("datasets/All_french_tweet_data.csv")
        df = df[["Tweet, Text", "User, name", "User, Description"]]
        df = df.rename(columns={"Tweet, Text": "Text", "User, name": "User"})
        df["Mixed features"] = df["Text"] + df["User"] + df["User, Description"]
        df = df[~df["Text"].str.startswith("RT", na=False)].copy()
        df = df.sample(n=10000, random_state=42)
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df["Mixed features"] = df["Mixed features"].apply(self.clean_up_text)
        df.to_csv("ty.csv", index=False)
        return df

class SchemaA1Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "In-Group", "Out-group", "Initiating Problem", "Intolerance", "Superiority of in-group",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Setting", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "2024_08_SCHEMA_A1"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/2024_08_SCHEMA_A1.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class OldSchemaA1Dataset(PreprocessedDataset):
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "2024_08_SCHEMA_A1"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/2024_08_SCHEMA_A1.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text"])
        df = df.reset_index()
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df

class SelectedDataset(PreprocessedDataset):
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name = "300*3Tweets_ARENAS"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self):
        df1 = pd.read_csv("datasets/300Tweets_ARENAS_Gender.csv")
        df2 = pd.read_csv("datasets/300Tweets_ARENAS_Nation.csv")
        df3 = pd.read_csv("datasets/300Tweets_ARENAS_Science.csv")
        df = pd.concat([df1, df2, df3], axis=0)
        df = df.reset_index(drop=True)
        df = df.rename(columns={"Document": "Text"})
        return df

# ---- FRENCH ----
class ARENASFrenchAnnotator1Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "In-Group", "Out-group", "Initiating Problem", "Intolerance", "Superiority of in-group",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instilment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Setting", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_FRENCH_1st_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_FRENCH_1st_Annotator.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_1st_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class ARENASFrenchAnnotator2Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = ARENASFrenchAnnotator1Dataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_FRENCH_2ND_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_FRENCH_2ND_Annotator.xlsx",
                           header=4, usecols="B, AH:BA")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_2nd_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return ARENASFrenchAnnotator1Dataset.get_context_attributes(self, indices, columns)

# ---- GERMAN ----
class ARENASGermanAnnotator1Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = ARENASFrenchAnnotator1Dataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_GERMAN_1st_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_GERMAN_1st_Annotator.xlsx",
                           header=4, usecols="B, AH:AZ")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_german_1st_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return ARENASFrenchAnnotator1Dataset.get_context_attributes(self, indices, columns)

class ARENASGermanAnnotator2Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = ARENASFrenchAnnotator1Dataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_GERMAN_2ND_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_GERMAN_2nd_Annotator.xlsx",
                           header=4, usecols="B, AH:AZ")
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        df.to_csv("ty_german_2nd_annotator.csv")
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return ARENASFrenchAnnotator1Dataset.get_context_attributes(self, indices, columns)

# ---- CYPRIOT ----
class ARENASCypriotAnnotator1Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Character(s)", "Setting",
        "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability",
        "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_CYPRIOT_1st_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        # Try all header rows until you find the real header
        found = False
        for hdr in range(0, 15):
            df = pd.read_excel("./datasets/ARENAS_DATA_CYPRIOT_1st_Annotator.xlsx", header=hdr)
            colnames = list(df.columns)
            if "Tweet, text" in colnames and "In-Group" in colnames and "Out-group" in colnames:
                found = True
                print(f"USING header={hdr}")
                break
        if not found:
            print("DEBUG: Could not find header row with required columns. Here are some column samples:")
            for hdr in range(0, 15):
                df = pd.read_excel("./datasets/ARENAS_DATA_CYPRIOT_1st_Annotator.xlsx", header=hdr)
                print(f"header={hdr}: {list(df.columns)[:20]}")
            raise ValueError("Could not find header row with required columns. Please check your Excel file.")

        df = df.rename(columns={"Tweet, text": "Text"})
        select_cols = ["Text"] + self.CONTEXT_COLUMNS_FULL
        missing = [col for col in select_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            print("DEBUG: Check for typos or extra spaces in your Excel columns. Here are the missing:", missing)
            raise ValueError(f"Missing columns: {missing}. Columns are: {df.columns}")
        df = df[select_cols]
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        return [tuple(self.data.iloc[i][col] for col in columns) for i in indices]

class ARENASCypriotAnnotator2Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = ARENASCypriotAnnotator1Dataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_CYPRIOT_2nd_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        found = False
        for hdr in range(0, 15):
            df = pd.read_excel("./datasets/ARENAS_DATA_CYPRIOT_2nd_Annotator.xlsx", header=hdr)
            colnames = list(df.columns)
            if "Tweet, text" in colnames and "In-Group" in colnames and "Out-group" in colnames:
                found = True
                print(f"USING header={hdr}")
                break
        if not found:
            print("DEBUG: Could not find header row with required columns. Here are some column samples:")
            for hdr in range(0, 15):
                df = pd.read_excel("./datasets/ARENAS_DATA_CYPRIOT_2nd_Annotator.xlsx", header=hdr)
                print(f"header={hdr}: {list(df.columns)[:20]}")
            raise ValueError("Could not find header row with required columns. Please check your Excel file.")

        df = df.rename(columns={"Tweet, text": "Text"})
        select_cols = ["Text"] + self.CONTEXT_COLUMNS_FULL
        missing = [col for col in select_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            print("DEBUG: Check for typos or extra spaces in your Excel columns. Here are the missing:", missing)
            raise ValueError(f"Missing columns: {missing}. Columns are: {df.columns}")
        df = df[select_cols]
        df = df.dropna(subset=["Text", "In-Group", "Out-group"])
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return ARENASCypriotAnnotator1Dataset.get_context_attributes(self, indices, columns)

# ---- SLOVENE ----
class ARENASSloveneAnnotator1Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group", "Polarization/Othering", "Perceived Threat",
        "Character(s)", "Setting", "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ]
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_SLOVENE_1st_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)
    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_SLOVENE_1st_Annotator.xlsx", header=4, usecols="E, AJ:BB")
        df = df.rename(columns={"Tweet, Text": "Text"})
        required_cols = ["Text", "In-Group", "Out-group"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            raise ValueError(f"Missing Slovene columns: {missing}. Columns are: {df.columns}")
        df = df.dropna(subset=required_cols)
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df
    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.CONTEXT_COLUMNS_FULL
        return [tuple(self.data.iloc[i][col] for col in columns) for i in indices]

class ARENASSloveneAnnotator2Dataset(PreprocessedDataset):
    CONTEXT_COLUMNS_FULL = ARENASSloveneAnnotator1Dataset.CONTEXT_COLUMNS_FULL
    def __init__(self, experiment_nb: int = 1, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name: str = "ARENAS_DATA_SLOVENE_2nd_Annotator"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)
    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_excel("./datasets/ARENAS_DATA_SLOVENE_2nd_Annotator.xlsx", header=4, usecols="E, AJ:BB")
        df = df.rename(columns={"Tweet, Text": "Text"})
        required_cols = ["Text", "In-Group", "Out-group"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print("DEBUG COLUMNS:", df.columns)
            raise ValueError(f"Missing Slovene columns: {missing}. Columns are: {df.columns}")
        df = df.dropna(subset=required_cols)
        df = df.reset_index(drop=True)
        df["Text"] = df["Text"].apply(self.clean_up_text)
        return df
    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        return ARENASSloveneAnnotator1Dataset.get_context_attributes(self, indices, columns)

# ---- TOXIGEN, LGBTEn, MigrantsEn as before ----
class ToxigenDataset(PreprocessedDataset):
    DEFAULT_CONTEXT_COLUMNS = [
        "target_group", "stereotype", "intent", "problem_group", "toxicity_annotator", "actual_method"
    ]
    def __init__(self, experiment_nb: int = 1, csv_path: Optional[str] = None, embeddings_path: Optional[str] = None, skip_embeddings: bool = False) -> None:
        self.dataset_name = "Toxigen"
        self.csv_path = csv_path or "datasets/Toxigen.csv"
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb, embeddings_path, skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        if "text" in df.columns:
            df["text"] = df["text"].apply(self.clean_up_text)
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        if columns is None:
            columns = self.DEFAULT_CONTEXT_COLUMNS
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class LGBTEnDataset(PreprocessedDataset):
    def __init__(
        self,
        experiment_nb: int = 1,
        embeddings_path: Optional[str] = None,
        skip_embeddings: bool = False,
        csv_path: Optional[str] = None
    ):
        self.dataset_name = "LGBTEn"
        self.csv_path = csv_path or "datasets/LGBTEn.csv"
        self.label_col = "annotation_type"
        self.context_cols = [
            "title", "post", "url", "source", "timestamp", "user_name", "user_id",
            "comment_url", "comment_timestamp", "annotation_type", "annotation_target", "annotation_annotator"
        ]
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb=experiment_nb, embeddings_path=embeddings_path, skip_embeddings=skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "post" in df.columns:
            df["post"] = df["post"].apply(self.clean_up_text)
        if "title" in df.columns:
            df["title"] = df["title"].apply(self.clean_up_text)
        if self.label_col in df.columns:
            labels_raw = df[self.label_col].tolist()
            if any(isinstance(lbl, str) for lbl in labels_raw):
                encoder = LabelEncoder()
                self.labels = encoder.fit_transform(labels_raw)
                self.label_encoder = encoder
            else:
                self.labels = labels_raw
        else:
            self.labels = []
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        columns = columns or self.context_cols
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out

class MigrantsEnDataset(PreprocessedDataset):
    def __init__(
        self,
        experiment_nb: int = 1,
        embeddings_path: Optional[str] = None,
        skip_embeddings: bool = False,
        csv_path: Optional[str] = None
    ):
        self.dataset_name = "MigrantsEn"
        self.csv_path = csv_path or "datasets/MigrantsEn.csv"
        self.label_col = "annotation_type"
        self.context_cols = [
            "title", "post", "url", "source", "timestamp", "user_name", "user_id",
            "comment_url", "comment_timestamp", "annotation_type", "annotation_target", "annotation_annotator"
        ]
        self.data = self.clean_up_dataset()
        super().__init__(experiment_nb=experiment_nb, embeddings_path=embeddings_path, skip_embeddings=skip_embeddings)

    def clean_up_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "post" in df.columns:
            df["post"] = df["post"].apply(self.clean_up_text)
        if "title" in df.columns:
            df["title"] = df["title"].apply(self.clean_up_text)
        if self.label_col in df.columns:
            labels_raw = df[self.label_col].tolist()
            if any(isinstance(lbl, str) for lbl in labels_raw):
                encoder = LabelEncoder()
                self.labels = encoder.fit_transform(labels_raw)
                self.label_encoder = encoder
            else:
                self.labels = labels_raw
        else:
            self.labels = []
        return df

    def get_context_attributes(self, indices: List[int], columns: Optional[List[str]] = None):
        columns = columns or self.context_cols
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Requested context columns not found in dataset: {missing_cols}")
        out = []
        for i in indices:
            entry = []
            for col in columns:
                entry.append(self.data.iloc[i][col])
            out.append(tuple(entry))
        return out