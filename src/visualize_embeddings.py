import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import argparse
import os
import nltk
from nltk.corpus import stopwords
import string

# Dataset imports
from preprocessing import (
    SchemaA1Dataset,
    FullFrenchTweetDataset,
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

sns.set(style="whitegrid", font_scale=1.2)

CUSTOM_FRENCH_STOPWORDS = set([
    "a", "faire", "fait", "c", "j", "l", "d", "n", "s", "t", "qu", "au", "aux", "du", "ce", "ces",
    "ma", "mes", "mon", "ta", "tes", "ton", "sa", "ses", "son", "leur", "leurs",
    "notre", "nos", "votre", "vos", "moi", "toi", "elle", "nous", "vous", "ils", "elles",
    "on", "se", "y", "en", "tout", "toute", "toutes", "tous"
])

DATASETS = {
    "SchemaA1": SchemaA1Dataset,
    "FullFrenchTweet": FullFrenchTweetDataset,
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

def get_stopwords(lang, use_custom_french):
    try:
        sw = set(stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords')
        sw = set(stopwords.words(lang))
    sw = set(w.lower() for w in sw)
    if use_custom_french and lang == "french":
        sw.update(w.lower() for w in CUSTOM_FRENCH_STOPWORDS)
    return sw

def tokenize(text):
    return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]

def get_top_words(texts, num_words=10, lang='english', use_custom_french=True):
    STOPWORDS = get_stopwords(lang, use_custom_french)
    words = []
    for text in texts:
        for w in tokenize(text):
            if w and w not in STOPWORDS and w.isalpha():
                words.append(w)
    return [w for w, _ in Counter(words).most_common(num_words)]

def load_node_texts(dataset_name, experiment_nb):
    cls = DATASETS.get(dataset_name)
    if cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset = cls(experiment_nb=experiment_nb)
    df = dataset.data
    if "Text" in df.columns:
        df = df.dropna(subset=["Text"]).reset_index(drop=True)
        return df["Text"].tolist(), dataset
    elif "text" in df.columns:
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        return df["text"].tolist(), dataset
    else:
        raise ValueError("No valid text column found.")

def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings and optionally extract cluster words.")
    parser.add_argument('--embeddings_file', type=str, required=True, help='Path to the embeddings file (.npy or .pt)')
    parser.add_argument('--output', type=str, default='embedding_vis.png', help='Path to save the visualization')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'], help='Dimensionality reduction method')
    parser.add_argument('--get_cluster_words', action='store_true', help='Output top words for each cluster')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--experiment_nb', type=int, default=1, help='Experiment number to load the right dataset')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--num_words', type=int, default=10, help='Number of top words per cluster')
    parser.add_argument('--stopwords_lang', type=str, default='english', help='Language for NLTK stopwords (e.g., english, french)')
    parser.add_argument('--use_custom_french_stopwords', action='store_true', help='Use custom French stopwords in addition to NLTK (only applies if --stopwords_lang french)')
    parser.add_argument('--save_cluster_words', action='store_true', help='Save cluster top words to a text file')
    parser.add_argument('--cluster_words_file', type=str, default='cluster_top_words.txt', help='Filename to save cluster words')
    parser.add_argument('--jitter', action='store_true', help='Add random jitter to points to reveal overlaps')
    parser.add_argument('--context_columns', nargs='+', default=None, help="Context columns to use for coloring/clustering (quote if contains spaces)")
    args = parser.parse_args()

    ext = os.path.splitext(args.embeddings_file)[-1]
    if ext in ['.pt', '.pth']:
        embeddings = torch.load(args.embeddings_file)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
    else:
        embeddings = np.load(args.embeddings_file)
    print(f"Loaded embeddings from {args.embeddings_file} with shape {embeddings.shape}")

    if args.method == 'tsne':
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        title = "t-SNE Visualization of Embeddings"
    else:
        X_embedded = PCA(n_components=2).fit_transform(embeddings)
        title = "PCA Visualization of Embeddings"

    rounded = np.round(X_embedded, 4)
    unique_positions = set(map(tuple, rounded))
    print(f"Total points: {len(X_embedded)}")
    print(f"Unique 2D positions after rounding: {len(unique_positions)}")
    if len(unique_positions) < len(X_embedded):
        print(f"WARNING: {len(X_embedded) - len(unique_positions)} points overlap in 2D projection.")

    colors = sns.color_palette("husl", args.num_clusters)

    if args.jitter:
        jitter = np.random.normal(0, 0.01, X_embedded.shape)
        X_embedded = X_embedded + jitter

    if args.get_cluster_words:
        if args.dataset is None:
            print("Error: --dataset must be provided if --get_cluster_words is set.")
            return
        texts, dataset_obj = load_node_texts(args.dataset, args.experiment_nb)
        if len(texts) != embeddings.shape[0]:
            print(f"Error: Number of texts ({len(texts)}) does not match number of embeddings ({embeddings.shape[0]}).")
            return
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=42).fit(embeddings)
        clusters = kmeans.labels_

        context_labels = None
        if args.context_columns is not None and hasattr(dataset_obj, "get_context_attributes"):
            context_labels = dataset_obj.get_context_attributes(range(len(texts)), columns=args.context_columns)

        plt.figure(figsize=(10, 10))
        cluster_lines = []
        for c in range(args.num_clusters):
            idxs = np.where(clusters == c)[0]
            plt.scatter(X_embedded[idxs, 0], X_embedded[idxs, 1], s=40, alpha=0.3, color=colors[c], label=f"Cluster {c}")

            cluster_texts = [texts[i] for i in idxs]
            top_words = get_top_words(
                cluster_texts,
                num_words=args.num_words,
                lang=args.stopwords_lang,
                use_custom_french=args.use_custom_french_stopwords
            )
            line = f"Cluster {c}: {top_words}"
            cluster_lines.append(line)

            centroid = X_embedded[idxs].mean(axis=0)
            if context_labels is not None:
                label_text = "\n".join(top_words) + "\n" + "\n".join([str(context_labels[i]) for i in idxs[:1]])
            else:
                label_text = "\n".join(top_words)
            plt.text(centroid[0], centroid[1], label_text, fontsize=14, fontweight='bold',
                     color=colors[c], bbox=dict(facecolor='white', alpha=0.7, pad=4),
                     horizontalalignment='center', verticalalignment='center')

        plt.title(f"{title} with Cluster Top Words")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(markerscale=1.5, fontsize=12)
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Visualization with cluster words saved to {args.output}")
        plt.show()

        if args.save_cluster_words:
            with open(args.cluster_words_file, "w", encoding="utf-8") as f:
                for line in cluster_lines:
                    f.write(line + "\n")
            print(f"Cluster top words saved to {args.cluster_words_file}")

    else:
        plt.figure(figsize=(8, 8))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap='tab10', alpha=0.3)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(args.output)
        print(f"Visualization saved to {args.output}")
        plt.show()

if __name__ == "__main__":
    main()