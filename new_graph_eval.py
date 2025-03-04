import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import FullFrenchTweetDataset, SchemaA1Dataset, SelectedDataset
from abc import ABC, abstractmethod


class GraphAnalysis(ABC):

    save_folder = "/home/cytech/Work/Master/Plots projet recherche/"

    def __init__(self, adjacency_file, original_dataset):
        with open(adjacency_file, 'rb') as f:
            adj_matrix = pickle.load(f)

        self.dataset = original_dataset
        self.node_features = self.dataset.data
        self.embeddings = self.dataset.embeddings
        # Verify alignment between adjacency matrix and features from original dataset
        assert adj_matrix.shape[0] == len(
            self.node_features), "Mismatch between matrix size and feature count"

        self.adj_matrix = adj_matrix.numpy()
        print("Loaded data")

    def create_nx_graph(self, threshold):
        adj_matrix_thresholded = self.adj_matrix.copy()
        adj_matrix_thresholded[adj_matrix_thresholded < threshold] = 0
        np.fill_diagonal(adj_matrix_thresholded, 0)

        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adj_matrix_thresholded)
        # Add node features to graph
        feature_dict = self.node_features.to_dict('index')
        nx.set_node_attributes(G, feature_dict)
        return G


class SingleGraphAnalysis(GraphAnalysis):

    def __init__(self, adjacency_file, original_dataset, threshold):
        super().__init__(adjacency_file, original_dataset)
        self.graph = self.create_nx_graph(threshold)

    def plot_graph(self, feature_to_plot):
        # Plot the graph with feature-based coloring
        plt.figure(figsize=(12, 8))
        codes, _ = pd.factorize(
            self.node_features[feature_to_plot])
        # Create layout and draw
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_color=codes,
                               cmap=plt.cm.viridis, node_size=50)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.1)
        plt.show()

    def plot_connected_components(self):
        """
        Plots an nx graph with nodes colored by their connected components.
        """
        # Find all connected components
        components = list(nx.connected_components(self.graph))

        # Create mapping from node to component index
        node_to_component = {}
        for idx, component in enumerate(components):
            for node in component:
                node_to_component[node] = idx

        # Create color map and labels
        colors = [node_to_component[node] for node in self.graph.nodes()]
        # Component index as text
        labels = {node: str(node_to_component[node])
                  for node in self.graph.nodes()}

        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=colors, cmap=plt.cm.tab20)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels, font_color='black')

        plt.axis('off')
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_connected_plot.png")
        plt.show()

    def retrieve_nodes_of_connected_comp(self, conn_comp_id):
        # Find all connected components
        components = list(nx.connected_components(self.graph))

        # Create mapping from node to component index
        nodes = []
        for idx, component in enumerate(components):
            if idx == conn_comp_id:
                for node in component:
                    nodes.append(node)
        self.node_features.iloc[nodes].to_csv("test.csv")

    def _calculate_component_group_ratios(self):
        """Calculate In-Group/Out-group pair ratios for each connected component."""
        components = list(nx.connected_components(self.graph))

        ratio_results = {}

        for comp_id, component in enumerate(components):
            # Get node indices for this component
            node_indices = list(component)

            # Get group pairs for these nodes
            # component_pairs = self.node_features.iloc[node_indices][[
            #     "In-Group", "Out-group"]]
            component_pairs = self.node_features.iloc[node_indices]["Topic"]

            # Calculate ratios
            pair_counts = component_pairs.value_counts()
            pair_ratios = pair_counts / pair_counts.sum()

            ratio_results[comp_id] = {
                'component_size': len(component),
                'pair_ratios': pair_ratios.to_dict()
            }
        print(ratio_results)
        return ratio_results

    def display_group_ratios(self):
        """Pretty-print the group ratio results."""
        ratio_results = self._calculate_component_group_ratios()
        for comp_id, data in ratio_results.items():
            print(f"Component {comp_id} (Size: {data['component_size']})")
            for (in_group, out_group), ratio in data['pair_ratios'].items():
                print(f"  {in_group}/{out_group}: {ratio:.2%}")
            print("\n" + "="*60 + "\n")


class Threshold_Analysis(GraphAnalysis):
    def __init__(self, adjacency_file, original_dataset, thresholds):
        super().__init__(adjacency_file, original_dataset)
        self.thresholds = thresholds

    def plot_nb_edges_variation(self):
        nb_edges = []
        for threshold in self.thresholds:
            graph = self.create_nx_graph(threshold)
            nb_edges.append(len(graph.edges))

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, nb_edges,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Number of edges depending on threshold')
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of edges')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_edges_per_threshold.png")
        plt.show()

    def plot_homophily_variation(self, feature_name):
        homophily_results = []
        # Calculate homophily for each threshold
        for threshold in self.thresholds:
            graph = self.create_nx_graph(threshold)
            current_homophily = nx.attribute_assortativity_coefficient(
                graph, feature_name)
            homophily_results.append(current_homophily)

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, homophily_results,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Homophily Variation for {feature_name}')
        plt.xlabel('Threshold Value')
        plt.ylabel('Homophily')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_homophily_{feature_name}.png")
        plt.show()

    def plot_connected_components_evolution(self):
        print("Calculating connected components")
        nb_connected = []
        for threshold in self.thresholds:
            print(f"Threshold : {threshold}")
            graph = self.create_nx_graph(threshold)
            nb_connected.append(nx.number_connected_components(graph))

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, nb_connected,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Number of connected components depending on threshold')
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of connected components')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_connected_per_threshold.png")
        plt.show()

    def plot_evolution_of_similarity_homophily(self, similarity_threshold):
        homophilies = []
        for threshold in self.thresholds:
            print(f"Threshold : {threshold}")
            graph = self.create_nx_graph(threshold)
            adjacency = nx.adjacency_matrix(graph).todense()
            current_homophily = self.similarity_homophily(
                adjacency, similarity_threshold)
            homophilies.append(current_homophily)

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, homophilies,
                 marker='o', linestyle='-', color='b')
        plt.title(
            f'Similarity homophily depending on edge threshold ({similarity_threshold})')
        plt.xlabel('Threshold Value')
        plt.ylabel('Similarity Homophily')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_simhomophily_{similarity_threshold}.png")
        plt.show()

    def plot_evolution_of_simple_homophily(self, feature):
        homophilies = []
        for threshold in self.thresholds:
            print(f"Threshold : {threshold}")
            graph = self.create_nx_graph(threshold)
            current_homophily = edge_homophily(graph, feature)
            homophilies.append(current_homophily)

        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, homophilies,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Simple homophily depending on edge threshold')
        plt.xlabel('Threshold Value')
        plt.ylabel('Simple Homophily')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_simplehomophily_{feature}.png")
        plt.show()

    def similarity_homophily(self, adj, similarity_threshold):
        count_above_threshold = 0
        edge_count = 0

        # Assuming the graph is undirected, we only need to check one triangle above the diagonal.
        n = adj.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    edge_count += 1
                    sim = cosine_similarity(
                        self.embeddings[i], self.embeddings[j])
                    if sim > similarity_threshold:
                        count_above_threshold += 1

        # Avoid division by zero if there are no edges.
        if edge_count == 0:
            return 0.0
        return count_above_threshold / edge_count


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def edge_homophily(G, feature):
    same_group_edges = 0
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0  # Avoid division by zero if there are no edges

    for u, v in G.edges():
        if G.nodes[u].get(feature) == G.nodes[v].get(feature):
            same_group_edges += 1

    homophily_index = same_group_edges / total_edges
    return homophily_index


def main():

    dataset = SchemaA1Dataset(experiment_nb=3)
    adjacency_file = './adjacency_matrices/adjacency_learned_epoch_1000_exp3.pkl'

    thresholds = np.linspace(0.0001, 0.5, 40)
    threshold_analysis = Threshold_Analysis(
        adjacency_file, dataset, thresholds)
    analysis = SingleGraphAnalysis(adjacency_file, dataset, threshold=0.115)
    print("Calculating connected components")
    threshold_analysis.plot_nb_edges_variation()
    threshold_analysis.plot_connected_components_evolution()
    threshold_analysis.plot_evolution_of_similarity_homophily(0.7)
    threshold_analysis.plot_evolution_of_simple_homophily("Topic")
    analysis.plot_connected_components()
    # analysis.display_group_ratios()


main()
