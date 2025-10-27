import os
import pandas as pd
import numpy as np
import re, emoji
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from cleanlab.classification import CleanLearning

DATASET_PATHS = {
    "Cypriot_1": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_CYPRIOT_1st_Annotator.xlsx",
    "Cypriot_2": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_CYPRIOT_2nd_Annotator.xlsx",
    "French_1": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_FRENCH_1st_Annotator.xlsx",
    "French_2": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_FRENCH_2ND_Annotator.xlsx",
    "German_1": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_GERMAN_1st_Annotator.xlsx",
    "German_2": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_GERMAN_2nd_Annotator.xlsx",
    "Slovene_1": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_SLOVENE_1st_Annotator.xlsx",
    "Slovene_2": "/home/dimitra/Documents/Github/ARENAS-Context-Aware-Graph-Structure-Learning-ContextGSL-/datasets/ARENAS_DATA_SLOVENE_2nd_Annotator.xlsx",
}

LABEL_COLUMNS = {
    "Cypriot": [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Character(s)", "Setting",
        "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability",
        "Conspiracy Theories", "Irony/Humor"
    ],
    "French": [
        "Topic", "In-Group", "Out-group", "Initiating Problem", "Intolerance", "Superiority of in-group",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instilment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Setting", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability", "Conspiracy Theories", "Irony/Humor"
    ],
    "German": [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator", "Intolerance",
        "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)",
        "Polarization/Othering", "Perceived Threat", "Character(s)", "Setting",
        "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability",
        "Conspiracy Theories", "Irony/Humor"
    ],
    "Slovene": [
        "Topic", "Tone of Post", "In-Group", "Out-group", "Narrator",
        "Intolerance", "Hostility to out-group", "Polarization/Othering", "Perceived Threat",
        "Character(s)", "Setting", "Initiating Problem", "Emotional response", "Solution",
        "Appeal to Authority", "Appeal to Reason", "Appeal to Probability",
        "Conspiracy Theories", "Irony/Humor"
    ]
}

def clean_up_text(text):
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

# --- Strict loader per language ---
def load_cypriot(path):
    df = pd.read_excel(path, header=4)
    df = df.rename(columns={"Tweet, text": "Text"})
    df["Text"] = df["Text"].apply(clean_up_text)
    return df

def load_french(path):
    df = pd.read_excel(path, header=4, usecols="B, AH:BA")
    df["Text"] = df["Text"].apply(clean_up_text)
    return df

def load_german(path):
    df = pd.read_excel(path, header=4, usecols="B, AH:AZ")
    df["Text"] = df["Text"].apply(clean_up_text)
    return df

def load_slovene(path):
    df = pd.read_excel(path, header=4, usecols="E, AJ:BB")
    df = df.rename(columns={"Tweet, Text": "Text"})
    df["Text"] = df["Text"].apply(clean_up_text)
    return df

def load_dataset(lang, annotator):
    key = f"{lang}_{annotator}"
    path = DATASET_PATHS[key]
    if lang == "Cypriot":
        return load_cypriot(path)
    elif lang == "French":
        return load_french(path)
    elif lang == "German":
        return load_german(path)
    elif lang == "Slovene":
        return load_slovene(path)
    else:
        raise ValueError(f"Unknown language: {lang}")

def run_cleanlab_label_issues(df, label_col):
    if label_col not in df.columns:
        return None
    valid_idx = df[label_col].notnull()
    texts = df.loc[valid_idx, "Text"].tolist()
    labels = df.loc[valid_idx, label_col].astype(str).tolist()
    if len(set(labels)) <= 1 or not texts:
        return None
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    transformer = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    embeddings = transformer.encode(texts, show_progress_bar=True)
    model = LogisticRegression(max_iter=1000)
    cl = CleanLearning(model, cv_n_folds=5)
    label_issues = cl.find_label_issues(X=embeddings, labels=y)
    label_issues["raw_text"] = np.array(texts)
    label_issues["given_label"] = np.array(labels)
    label_issues["predicted_label_name"] = encoder.inverse_transform(label_issues["predicted_label"])
    return label_issues

def mean_label_quality(label_issues, given_label_col='given_label', confidence_col='label_quality'):
    class_conf = {}
    for class_value in label_issues[given_label_col].unique():
        mask = label_issues[given_label_col] == class_value
        mean_conf = label_issues.loc[mask, confidence_col].mean()
        class_conf[class_value] = mean_conf
    return class_conf

# --- Annotator comparison loop ---
results_rows = []
summary_wins = {}
for lang in ["Cypriot", "French", "German", "Slovene"]:
    df1 = load_dataset(lang, 1)
    df2 = load_dataset(lang, 2)
    colset1 = set(df1.columns)
    colset2 = set(df2.columns)
    common_cols = set(LABEL_COLUMNS[lang]) & colset1 & colset2
    wins_1 = wins_2 = ties = 0
    for label_col in common_cols:
        issues1 = run_cleanlab_label_issues(df1, label_col)
        issues2 = run_cleanlab_label_issues(df2, label_col)
        if issues1 is None or issues2 is None:
            continue
        conf1 = mean_label_quality(issues1)
        conf2 = mean_label_quality(issues2)
        for class_value in set(conf1.keys()) | set(conf2.keys()):
            c1 = conf1.get(class_value, np.nan)
            c2 = conf2.get(class_value, np.nan)
            if np.isnan(c1) and np.isnan(c2):
                continue
            if not np.isnan(c1) and not np.isnan(c2):
                if np.isclose(c1, c2):
                    winner = "Tie"
                    ties += 1
                elif c1 > c2:
                    winner = "Annotator 1"
                    wins_1 += 1
                else:
                    winner = "Annotator 2"
                    wins_2 += 1
            elif not np.isnan(c1):
                winner = "Annotator 1"
                wins_1 += 1
            elif not np.isnan(c2):
                winner = "Annotator 2"
                wins_2 += 1
            results_rows.append({
                "Language": lang,
                "Label Column": label_col,
                "Class": class_value,
                "Annotator 1 Mean Confidence": c1,
                "Annotator 2 Mean Confidence": c2,
                "Winner": winner
            })
    # Summary per dataset
    if wins_1 > wins_2:
        dataset_winner = "Annotator 1"
    elif wins_2 > wins_1:
        dataset_winner = "Annotator 2"
    else:
        dataset_winner = "Tie"
    summary_wins[lang] = {"Annotator 1": wins_1, "Annotator 2": wins_2, "Ties": ties, "Dataset Winner": dataset_winner}

results_df = pd.DataFrame(results_rows)
print("\n=== Cleanlab Annotator Confidence Comparison ===")
print(results_df)
results_df.to_csv("cleanlab_annotator_confidence_comparison_full.csv", index=False)

# --- Summary table ---
summary_rows = []
for lang, stats in summary_wins.items():
    summary_rows.append({
        "Language": lang,
        "Annotator 1 Wins": stats["Annotator 1"],
        "Annotator 2 Wins": stats["Annotator 2"],
        "Ties": stats["Ties"],
        "Dataset Winner": stats["Dataset Winner"]
    })
summary_df = pd.DataFrame(summary_rows)
print("\n=== Per-Dataset Annotator Win Summary ===")
print(summary_df)
summary_df.to_csv("cleanlab_annotator_win_summary.csv", index=False)

# --- Overall winner ---
overall_counts = {"Annotator 1": 0, "Annotator 2": 0, "Tie": 0}
for row in summary_rows:
    overall_counts[row["Dataset Winner"]] += 1
overall_summary = pd.DataFrame([{
    "Annotator 1 Datasets Won": overall_counts["Annotator 1"],
    "Annotator 2 Datasets Won": overall_counts["Annotator 2"],
    "Tied Datasets": overall_counts["Tie"],
    "Overall Winner": max(overall_counts, key=lambda k: overall_counts[k] if k != "Tie" else -1)
}])
print("\n=== Overall Annotator Winner ===")
print(overall_summary)
overall_summary.to_csv("cleanlab_annotator_overall_winner.csv", index=False)
