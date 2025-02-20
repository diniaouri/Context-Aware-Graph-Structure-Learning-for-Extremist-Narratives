import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GraphAnalysis():

    def __init__(self):
        # Load adjacency matrix from pickle file
        with open('./adjacency_matrices/adjacency_learned_epoch_1000_exp3.pkl', 'rb') as f:
            adj_matrix = pickle.load(f)

        # Load node features from dataframe (assuming CSV for example)
        self.node_features = pd.read_excel("2024_08_SCHEMA_A1.xlsx",
                                           header=4, usecols="B, AH:BA")
        self.node_features = self.node_features.dropna(
            subset=["Text", "In-Group", "Out-group"])
        self.node_features = self.node_features.reset_index(drop=True)

        # Verify alignment between matrix and features
        assert adj_matrix.shape[0] == len(
            self.node_features), "Mismatch between matrix size and feature count"

        self.adj_matrix = adj_matrix.numpy()

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

    def plot_graph(self, feature_to_plot, threshold):

        # Plot the graph with feature-based coloring
        graph = self.create_nx_graph(threshold)
        plt.figure(figsize=(12, 8))
        colors = self.node_features[feature_to_plot]
        codes, uniques = pd.factorize(
            self.node_features[feature_to_plot])
        # Create layout and draw
        pos = nx.spring_layout(graph, seed=42)
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=codes,
                                       cmap=plt.cm.viridis, node_size=50)
        edges = nx.draw_networkx_edges(graph, pos, alpha=0.1)

        # Add colorbar if feature is continuous
        if pd.api.types.is_numeric_dtype(colors):
            plt.colorbar(nodes, label=feature_to_plot)
        plt.show()

    def homophily_calc(self, graph):
        feature_homophily = {}
        for feature_name in self.node_features.columns:
            # Calculate homophily for each feature
            feature_homophily[feature_name] = nx.attribute_assortativity_coefficient(
                graph, feature_name)
        return feature_homophily

    def analysis_of_threshold_on_homophily(self, thresholds):
        # Initialize dictionary to store results
        homophily_results = {feature: []
                             for feature in self.node_features.columns}
        # Calculate homophily for each threshold
        for threshold in thresholds:
            graph = self.create_nx_graph(threshold)
            current_homophily = self.homophily_calc(graph)

            # Store results for each feature
            for feature, value in current_homophily.items():
                homophily_results[feature].append(value)

        return homophily_results

    def plot_nb_edges_variation(self, thresholds):
        nb_edges = []
        for threshold in thresholds:
            graph = self.create_nx_graph(threshold)
            nb_edges.append(len(graph.edges))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, nb_edges,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Number of edges depending on threshold')
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of edges')
        plt.grid(True)
        plt.show()

    def plot_homophily_variation(self, feature_name, thresholds):
        homophily_results = self.analysis_of_threshold_on_homophily(thresholds)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, homophily_results[feature_name],
                 marker='o', linestyle='-', color='b')
        plt.title(f'Homophily Variation for {feature_name}')
        plt.xlabel('Threshold Value')
        plt.ylabel('Homophily')
        plt.grid(True)
        plt.show()

    def plot_connected_components_evolution(self, thresholds):
        nb_connected = []
        for threshold in thresholds:
            graph = self.create_nx_graph(threshold)
            nb_connected.append(nx.number_connected_components(graph))

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, nb_connected,
                 marker='o', linestyle='-', color='b')
        plt.title(f'Number of connected components depending on threshold')
        plt.xlabel('Threshold Value')
        plt.ylabel('Number of connected components')
        plt.grid(True)
        plt.show()

    def plot_connected_components(self, threshold):
        """
        Plots an nx graph with nodes colored by their connected components.
        """
        # Find all connected components in the graph
        graph = self.create_nx_graph(threshold)
        # Find all connected components
        components = list(nx.connected_components(graph))

        # Create mapping from node to component index
        node_to_component = {}
        for idx, component in enumerate(components):
            for node in component:
                node_to_component[node] = idx

        # Create color map and labels
        colors = [node_to_component[node] for node in graph.nodes()]
        # Component index as text
        labels = {node: str(node_to_component[node]) for node in graph.nodes()}

        # Draw the graph
        pos = nx.spring_layout(graph)
        plt.figure(figsize=(10, 6))

        # Draw nodes with component colors
        nx.draw_networkx_nodes(
            graph, pos, node_color=colors, cmap=plt.cm.tab20)

        # Draw edges
        nx.draw_networkx_edges(graph, pos)

        # Draw component index labels
        nx.draw_networkx_labels(graph, pos, labels=labels, font_color='black')

        plt.axis('off')
        plt.show()

    def retrieve_nodes_of_connected_comp(self, conn_comp_id, threshold):
        graph = self.create_nx_graph(threshold)
        # Find all connected components
        components = list(nx.connected_components(graph))

        # Create mapping from node to component index
        nodes = []
        for idx, component in enumerate(components):
            if idx == conn_comp_id:
                for node in component:
                    nodes.append(node)
        self.node_features.iloc[nodes].to_csv("test.csv")

    def calculate_component_group_ratios(self, threshold):
        """Calculate In-Group/Out-group pair ratios for each connected component."""
        graph = self.create_nx_graph(threshold)
        components = list(nx.connected_components(graph))

        ratio_results = {}

        for comp_id, component in enumerate(components):
            # Get node indices for this component
            node_indices = list(component)

            # Get group pairs for these nodes
            component_pairs = self.node_features.iloc[node_indices][[
                "In-Group", "Out-group"]]

            # Calculate ratios
            pair_counts = component_pairs.value_counts()
            pair_ratios = pair_counts / pair_counts.sum()

            ratio_results[comp_id] = {
                'component_size': len(component),
                'pair_ratios': pair_ratios.to_dict()
            }

        return ratio_results

    def display_group_ratios(self, ratio_results):
        """Pretty-print the group ratio results."""
        for comp_id, data in ratio_results.items():
            print(f"Component {comp_id} (Size: {data['component_size']})")
            for (in_group, out_group), ratio in data['pair_ratios'].items():
                print(f"  {in_group}/{out_group}: {ratio:.2%}")
            print("\n" + "="*60 + "\n")


def main():

    analysis = GraphAnalysis()
    # thresholds = np.linspace(0.00001, 0.5, 400)
    # analysis.plot_nb_edges_variation(thresholds)
    # analysis.plot_connected_components_evolution(thresholds)
    # analysis.plot_homophily_variation("Topic", thresholds)
    analysis.plot_connected_components(threshold=0.115)
    ratio_results = analysis.calculate_component_group_ratios(0.115)
    analysis.display_group_ratios(ratio_results)
    analysis.retrieve_nodes_of_connected_comp(38, 0.115)


main()
