import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from preprocessing import FullFrenchTweetDataset, SchemaA1Dataset, SelectedDataset, OldSchemaA1Dataset
from networkx.algorithms.community import girvan_newman
import csv


class GraphAnalysis():

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

    def plot_communities(self, nb_communities):

        communities_gen = girvan_newman(self.graph)
        desired_communities = None
        for communities in communities_gen:
            if len(communities) >= nb_communities:
                desired_communities = communities
                break
        if desired_communities is None:
            raise ValueError(
                "Could not reach the desired number of communities.")

        # Create a community ID label dictionary (node: community_id)
        community_labels = {}
        for comm_idx, community in enumerate(desired_communities):
            for node in community:
                # Convert to string for cleaner display
                community_labels[node] = str(comm_idx)

        # Create colormap for communities
        color_map = []
        for node in self.graph.nodes():
            for comm_idx, comm in enumerate(desired_communities):
                if node in comm:
                    color_map.append(comm_idx)
                    break
        self._store_communities(desired_communities)
        # Plot graph
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(10, 6))

        # Draw nodes with community colors
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=color_map, cmap=plt.cm.tab20)

        # Draw edges (optional, for cleaner visualization)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2)

        # Draw community labels instead of node labels
        nx.draw_networkx_labels(
            self.graph, pos, labels=community_labels, font_size=10)

        plt.title(f"Community Structure ({len(desired_communities)} Groups)")
        plt.axis("off")
        plt.show()

    def community_homophily(self, nb_communities, features):
        communities_gen = girvan_newman(self.graph)
        desired_communities = None
        for communities in communities_gen:
            if len(communities) >= nb_communities:
                desired_communities = communities
                break
        if desired_communities is None:
            raise ValueError(
                "Could not reach the desired number of communities.")

        subG = nx.Graph()
        i = 0
        # Process each community separately
        maximums_of_distribution = []
        for community_nodes in desired_communities:
            i += 1
            # if len(community_nodes) > 5:
            component_pairs = self.node_features.iloc[list(community_nodes)][[
                "In-Group"]]
            pair_counts = component_pairs.value_counts()
            pair_ratios = pair_counts / pair_counts.sum()

            # print(f"Community {i} (len : {len(community_nodes)}) : ")
            # print(pair_ratios)
            if len(community_nodes) != 1:
                maximum_of_distrib = max(pair_ratios) * len(community_nodes)
            else:
                maximum_of_distrib = 0
            maximums_of_distribution.append(maximum_of_distrib)
            # Create a subgraph for the current community and copy it
            community_sub = self.graph.subgraph(community_nodes).copy()
            # Combine this community subgraph with the overall subgraph
            subG = nx.compose(subG, community_sub)
        print(
            f"Average maximum of distribution : {np.sum(maximums_of_distribution)/len(self.graph.nodes)}")
        other_graph_maximums = []
        for community in list(nx.connected_components(self.graph)):
            component_pairs = self.node_features.iloc[list(community)][[
                "Out-group"]]
            pair_counts = component_pairs.value_counts()
            pair_ratios = pair_counts / pair_counts.sum()
            if len(community) != 1:
                maximum_of_distrib = max(pair_ratios) * len(community)
            else:
                maximum_of_distrib = 0
            other_graph_maximums.append(maximum_of_distrib)
        print(
            f"Average maximum of distribution : {np.sum(other_graph_maximums)/len(self.graph.nodes)}")
        # Measures how much the communities are concentrated in one group. Bigger communities have more weight.
        for feature in features:
            homophily = nx.attribute_assortativity_coefficient(subG, feature)
            print(f"{feature} : {homophily}")

        color_map = []
        for node in subG.nodes():
            for comm_idx, comm in enumerate(desired_communities):
                if node in comm:
                    color_map.append(comm_idx)
                    break
        community_labels = {}
        for comm_idx, community in enumerate(desired_communities):
            for node in community:
                # Convert to string for cleaner display
                community_labels[node] = str(comm_idx)

        pos = nx.spring_layout(subG)
        plt.figure(figsize=(10, 6))

        # Draw nodes with community colors
        nx.draw_networkx_nodes(
            subG, pos, node_color=color_map, cmap=plt.cm.tab20)

        # Draw edges (optional, for cleaner visualization)
        nx.draw_networkx_edges(subG, pos, alpha=0.2)

        # Draw community labels instead of node labels
        nx.draw_networkx_labels(
            subG, pos, labels=community_labels, font_size=10)

        plt.title(f"Community Structure ({len(desired_communities)} Groups)")
        plt.axis("off")
        plt.show()

    def _store_communities(self, communities):
        attribute_keys = set()
        for node in self.graph.nodes:
            attribute_keys.update(self.graph.nodes[node].keys())
        attribute_keys = sorted(attribute_keys)

        # Define the specific features to include
        target_features = ["Out-group", "In-Group", "Topic", "Text"]
        headers = ["Community", "Node"] + target_features

        with open("communities.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for comm_idx, community in enumerate(communities):
                for node in sorted(community):  # Sort nodes for consistency
                    attrs = self.graph.nodes[node]
                    # Extract only the target features (use empty string if missing)
                    row = [comm_idx, node] + \
                        [attrs.get(feature, "") for feature in target_features]
                    writer.writerow(row)

    def _calculate_component_group_ratios(self):
        """Calculate In-Group/Out-group pair ratios for each connected component."""
        components = list(nx.connected_components(self.graph))

        ratio_results = {}

        for comp_id, component in enumerate(components):
            # Get node indices for this component
            node_indices = list(component)

            # Get group pairs for these nodes
            component_pairs = self.node_features.iloc[node_indices][[
                "In-Group", "Out-group"]]
            # component_pairs = self.node_features.iloc[node_indices]["Topic"]

            # Calculate ratios
            pair_counts = component_pairs.value_counts()
            pair_ratios = pair_counts / pair_counts.sum()

            ratio_results[comp_id] = {
                'component_size': len(component),
                'pair_ratios': pair_ratios.to_dict()
            }
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
        plt.title(f'Assortative mixing for {feature_name}')
        plt.xlabel('Threshold Value')
        plt.ylabel('Assortative Mixing')
        plt.grid(True)
        plt.savefig(GraphAnalysis.save_folder +
                    f"Exp{self.dataset.experiment}_homophily_{feature_name.replace('/', '')}.png")
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
    matplotlib.rcParams['font.size'] = 18  # Set global font size
    dataset = SchemaA1Dataset(experiment_nb=3)
    adjacency_file = './adjacency_matrices/adjacency_learned_epoch_1000_exp3.pkl'

    thresholds = np.linspace(0.0001, 0.5, 400)
    threshold_analysis = Threshold_Analysis(
        adjacency_file, dataset, thresholds)
    # Threshold 0.05 for girvan newman
    analysis = SingleGraphAnalysis(adjacency_file, dataset, threshold=0.05)
    # print("Calculating connected components")
    # threshold_analysis.plot_nb_edges_variation()
    print(analysis.embeddings.shape)
    # threshold_analysis.plot_connected_components_evolution()
    # threshold_analysis.plot_homophily_variation("Topic")
    # threshold_analysis.plot_homophily_variation("In-Group")
    # threshold_analysis.plot_homophily_variation("Out-group")
    # threshold_analysis.plot_homophily_variation("Superiority of in-group")
    # threshold_analysis.plot_homophily_variation("Intolerance")
    # threshold_analysis.plot_homophily_variation("Polarization/Othering")

    # threshold_analysis.plot_evolution_of_similarity_homophily(0.7)
    # threshold_analysis.plot_evolution_of_simple_homophily("Topic")
    # analysis.plot_connected_components()
    # analysis.display_group_ratios()
    # analysis.retrieve_nodes_of_connected_comp(20)
    # for x in range(5, 50, 10):
    #     print(f"{x} :")
    #     analysis.community_homophily(x, ["Topic", "Out-group", "In-Group"])
    # analysis.community_homophily(35, ["Topic", "Out-group", "In-Group"])
    # analysis.plot_communities(5)
    # analysis.community_homophily(35, ["Topic"])
    # analysis.retrieve_nodes_of_connected_comp(6)


main()
