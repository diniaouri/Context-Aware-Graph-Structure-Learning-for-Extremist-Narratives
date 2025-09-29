import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import string
import os
import networkx as nx
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
import plotly.express as px

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

def get_stopwords(lang="english", use_custom_french=True):
    try:
        sw = set(stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords')
        sw = set(stopwords.words(lang))
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

def load_node_metadata(dataset_name, experiment_nb):
    cls = DATASETS.get(dataset_name)
    if cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    dataset = cls(experiment_nb=experiment_nb, embeddings_path=None, skip_embeddings=True)
    if hasattr(dataset, "dataset_name") and dataset.dataset_name == "Toxigen":
        df = dataset.data.dropna(subset=["text"]).reset_index(drop=True)
    elif "Text" in dataset.data.columns:
        df = dataset.data.dropna(subset=["Text"]).reset_index(drop=True)
    elif "text" in dataset.data.columns:
        df = dataset.data.dropna(subset=["text"]).reset_index(drop=True)
    else:
        df = dataset.data.reset_index(drop=True)
    return df, dataset

def adjacency_matrix_stats(adj_matrix, threshold=None):
    if threshold is not None:
        adj_matrix = adj_matrix.copy()
        adj_matrix[adj_matrix < threshold] = 0
    num_edges = np.count_nonzero(adj_matrix) - np.count_nonzero(np.diag(adj_matrix))
    density = num_edges / (adj_matrix.shape[0] ** 2 - adj_matrix.shape[0])
    max_val = np.max(adj_matrix)
    min_val = np.min(adj_matrix)
    mean_val = np.mean(adj_matrix)
    return {
        "Number of edges": int(num_edges),
        "Density": float(density),
        "Max value": float(max_val),
        "Min value": float(min_val),
        "Mean value": float(mean_val)
    }

def get_connected_components(graph):
    return list(nx.connected_components(graph))

def connected_component_sizes(graph):
    return [len(comp) for comp in get_connected_components(graph)]

def number_connected_components(graph):
    return nx.number_connected_components(graph)

def attribute_assortativity(graph, feature):
    try:
        return nx.attribute_assortativity_coefficient(graph, feature)
    except Exception:
        return None

def edge_homophily(graph, feature):
    same_group_edges = 0
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0
    for u, v in graph.edges():
        if graph.nodes[u].get(feature) == graph.nodes[v].get(feature):
            same_group_edges += 1
    return same_group_edges / total_edges if total_edges > 0 else 0

def basic_community_detection(graph, nb_communities=5):
    from networkx.algorithms.community import girvan_newman
    communities_gen = girvan_newman(graph)
    desired = None
    for coms in communities_gen:
        if len(coms) >= nb_communities:
            desired = coms
            break
    return desired

def get_top_words_for_community(node_features, community_nodes, num_words=10, lang='english', use_custom_french=True):
    if "text" in node_features.columns:
        texts = node_features.iloc[list(community_nodes)]["text"].tolist()
    elif "Text" in node_features.columns:
        texts = node_features.iloc[list(community_nodes)]["Text"].tolist()
    else:
        texts = []
    return get_top_words(texts, num_words=num_words, lang=lang, use_custom_french=use_custom_french)

def plot_interactive_community_graph(graph, node_features, communities, feature_to_show="topic"):
    pos = nx.spring_layout(graph, seed=42)
    node_x, node_y = [], []
    node_labels = []
    node_colors = []
    hover_texts = []
    community_labels = {}
    color_palette = [
        "rgba(56,108,176,0.8)", "rgba(127,205,187,0.8)", "rgba(255,255,51,0.8)",
        "rgba(65,182,196,0.8)", "rgba(161,218,180,0.8)", "rgba(255,127,0,0.8)",
        "rgba(240,59,32,0.8)", "rgba(186,189,182,0.8)", "rgba(141,211,199,0.8)",
        "rgba(255,255,179,0.8)", "rgba(190,186,218,0.8)", "rgba(251,128,114,0.8)"
    ]
    for idx, comm in enumerate(communities):
        for node in comm:
            community_labels[node] = idx

    for node in graph.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        comm_idx = community_labels.get(node, 0)
        node_colors.append(color_palette[comm_idx % len(color_palette)])
        node_labels.append(str(node))
        if "text" in node_features.columns and feature_to_show == "text":
            feature_val = node_features.loc[node, "text"]
        elif feature_to_show in node_features.columns:
            feature_val = node_features.loc[node, feature_to_show]
        else:
            feature_val = ""
        show_text = node_features.loc[node, "text"] if "text" in node_features.columns else node_features.loc[node, "Text"] if "Text" in node_features.columns else ""
        hover = f"Cluster: {comm_idx}<br>Node: {node}<br>{feature_to_show}: {feature_val}<br>Text: {show_text}"
        hover_texts.append(hover)

    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            color=node_colors,
            size=24,
            line_width=2
        ),
        text=node_labels,
        textposition="middle center",
        hovertext=hover_texts,
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Community Structure ({len(communities)} Groups)",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

def plot_cluster_feature_stats(node_features, community_nodes, feature_options):
    df = node_features.iloc[list(community_nodes)]
    st.markdown("### Cluster Feature Statistics")
    for feature in feature_options:
        if feature.lower() == "text":
            continue
        values = df[feature]
        value_counts = values.value_counts(normalize=True).sort_values(ascending=False) * 100
        fig = px.bar(x=value_counts.index.astype(str), y=value_counts.values,
                     labels={'x': feature, 'y': 'Percentage'}, title=f"{feature} breakdown in cluster")
        fig.update_layout(yaxis=dict(ticksuffix='%'), xaxis_title=feature)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            ", ".join([f"**{k}**: {v:.1f}%" for k, v in zip(value_counts.index, value_counts.values)])
        )
    st.markdown("---")

st.set_page_config(layout="wide")
st.title("Embeddings & Graph Visualization Dashboard")

embeddings_file = st.sidebar.file_uploader("Upload Embeddings (.npy or .pt)", type=["npy", "pt"])
adjacency_file = st.sidebar.file_uploader("Upload Adjacency Matrix (.pkl)", type=["pkl"])
dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
experiment_nb = st.sidebar.number_input("Experiment Number", min_value=1, value=1)
method = st.sidebar.selectbox("Dimensionality Reduction", ["t-SNE", "PCA"])
num_clusters = st.sidebar.slider("Number of Clusters", 2, 15, 5)
show_top_words = st.sidebar.checkbox("Show Top Words per Cluster", True)
num_top_words = st.sidebar.slider("Number of Top Words", 3, 20, 8)
perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 20)
stopwords_lang = st.sidebar.selectbox("Stopwords Language", ["english", "french"])
use_custom_french = st.sidebar.checkbox("Use Custom French Stopwords", True)

tab1, tab2 = st.tabs(["Embedding Visualization", "Graph Visualization"])

with tab1:
    if embeddings_file:
        ext = os.path.splitext(embeddings_file.name)[-1]
        if ext in ['.pt', '.pth']:
            import torch
            embeddings = torch.load(embeddings_file)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        else:
            embeddings = np.load(embeddings_file)
        st.write(f"Loaded embeddings shape: {embeddings.shape}")
        st.subheader(f"Embedding File: {embeddings_file.name}")

        try:
            node_features, dataset_obj = load_node_metadata(dataset_name, experiment_nb)
            if "text" in node_features.columns:
                texts = node_features["text"].tolist()
            else:
                texts = node_features["Text"].tolist()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            texts = None

        if texts and len(texts) != embeddings.shape[0]:
            st.warning(f"Number of texts ({len(texts)}) doesn't match number of embeddings ({embeddings.shape[0]}). Disabling top words.")
            show_top_words = False

        if method == "t-SNE":
            X_embedded = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
            title = "t-SNE Visualization"
        else:
            X_embedded = PCA(n_components=2).fit_transform(embeddings)
            title = "PCA Visualization"

        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        clusters = kmeans.labels_

        colors = plt.get_cmap("tab10")
        plt.figure(figsize=(10, 10))
        cluster_words = []
        for c in range(num_clusters):
            idxs = np.where(clusters == c)[0]
            plt.scatter(X_embedded[idxs, 0], X_embedded[idxs, 1], s=40, alpha=0.4, color=colors(c), label=f"Cluster {c}")

            if show_top_words and texts:
                cluster_texts = [texts[i] for i in idxs]
                top_words = get_top_words(cluster_texts, num_words=num_top_words, lang=stopwords_lang, use_custom_french=use_custom_french)
                centroid = X_embedded[idxs].mean(axis=0)
                label_text = "\n".join(top_words)
                plt.text(centroid[0], centroid[1], label_text, fontsize=13, fontweight='bold',
                        color=colors(c), bbox=dict(facecolor='white', alpha=0.7, pad=3),
                        horizontalalignment='center', verticalalignment='center')
                cluster_words.append(f"Cluster {c}: {top_words}")

        plt.title(title + " with Cluster Top Words" if show_top_words else title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        st.pyplot(plt)

        if show_top_words and cluster_words:
            st.subheader("Top Words per Cluster (Embeddings)")
            for line in cluster_words:
                st.write(line)

with tab2:
    if adjacency_file:
        import pickle
        adj_matrix = None
        try:
            with adjacency_file as f:
                adj_matrix = pickle.load(f)
        except Exception as e:
            st.error(f"Could not load adjacency matrix: {e}")

        if adj_matrix is not None:
            if hasattr(adj_matrix, "numpy"):
                adj_matrix = adj_matrix.numpy()
            st.write(f"Adjacency Matrix File: {adjacency_file.name}")
            st.write(f"Adjacency Matrix Shape: {adj_matrix.shape}")

            threshold = st.slider("Threshold (set to zero edges below)", min_value=float(np.min(adj_matrix)), max_value=float(np.max(adj_matrix)), value=float(np.min(adj_matrix)), step=0.001)

            try:
                node_features, dataset_obj = load_node_metadata(dataset_name, experiment_nb)
            except Exception:
                node_features = None

            feature_options = node_features.columns.tolist() if node_features is not None else []

            adj = adj_matrix.copy()
            if threshold is not None:
                adj[adj < threshold] = 0
            np.fill_diagonal(adj, 0)
            graph = nx.from_numpy_array(adj)
            if node_features is not None:
                feature_dict = node_features.to_dict('index')
                nx.set_node_attributes(graph, feature_dict)

            nb_communities = st.slider("Number of communities to detect", 2, min(20, graph.number_of_nodes()), 5)
            communities = None
            if graph.number_of_edges() > 0:
                communities = basic_community_detection(graph, nb_communities=nb_communities)

            stats = adjacency_matrix_stats(adj_matrix, threshold=threshold)
            st.write("Adjacency Matrix Statistics:")
            for k, v in stats.items():
                st.write(f"- {k}: {v}")

            st.subheader("Graph Evaluation")
            st.write(f"Number of connected components: {number_connected_components(graph)}")

            feature_eval = st.selectbox("Evaluate assortativity/homophily for feature", options=feature_options, index=0)
            if feature_eval and feature_eval != "None":
                assort = attribute_assortativity(graph, feature_eval)
                st.write(f"Attribute assortativity for {feature_eval}: {assort if assort is not None else 'N/A'}")
                homophily = edge_homophily(graph, feature_eval)
                st.write(f"Edge-based homophily for {feature_eval}: {homophily:.3f}")

            st.subheader("Connected Component Sizes")
            cc_sizes = connected_component_sizes(graph)
            st.write(f"Sizes (top 10): {sorted(cc_sizes, reverse=True)[:10]}")

            if communities is not None:
                st.subheader("Community Structure (Interactive)")
                feature_to_show = st.selectbox("Select node feature to show in hover", feature_options, index=0)
                fig = plot_interactive_community_graph(graph, node_features, communities, feature_to_show=feature_to_show)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Top Words per Community (Adjacency)")
                for idx, community_nodes in enumerate(communities):
                    top_words = get_top_words_for_community(
                        node_features, community_nodes,
                        num_words=num_top_words,
                        lang=stopwords_lang,
                        use_custom_french=use_custom_french
                    )
                    st.markdown(f"**Community {idx}**: {', '.join(top_words)}")

                st.subheader("View Sentences Per Community")
                selected_community = st.selectbox(
                    "Choose community to display sentences and see its statistics", 
                    list(range(len(communities)))
                )
                community_nodes = list(communities[selected_community])
                if "text" in node_features.columns:
                    sentences = node_features.iloc[community_nodes]["text"].tolist()
                else:
                    sentences = node_features.iloc[community_nodes]["Text"].tolist()
                st.write(f"Sentences in community {selected_community}:")
                for i, sentence in enumerate(sentences[:50]):
                    st.markdown(f"**{i+1}.** {sentence}")
                if len(sentences) > 50:
                    st.write(f"... and {len(sentences)-50} more")

                plot_cluster_feature_stats(node_features, community_nodes, feature_options)

                st.subheader("Inspect Node (All Details)")
                node_indices = list(graph.nodes())
                selected_node = st.selectbox("Select node to inspect", node_indices)
                if node_features is not None and selected_node is not None:
                    st.markdown(f"**Node Index:** {selected_node}")
                    node_row = node_features.iloc[selected_node]
                    for col in node_features.columns:
                        st.markdown(f"**{col}:** {node_row[col]}")
            else:
                st.write("Not enough edges for community detection.")