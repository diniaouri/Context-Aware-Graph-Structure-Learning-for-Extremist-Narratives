__version__ = "0.2.0"
__author__ = "rayane ghilene"

# from arenas_graph.config import Config
# from arenas_graph.utils.repro import seed_everything

# Core data utilities
from arenas_graph.data import (
    embed_texts,
    build_knn_graph,
    build_bipartite_text_group,
    load_clean_arenas_dataset,
)

# Core models & trainers
from arenas_graph.models.training import (
    train_node_classification,
    train_link_prediction,
)

__all__ = [
    # Metadata
    "__version__",
    "__author__",

    # Data utilities
    "embed_texts",
    "build_knn_graph",
    "build_bipartite_text_group",
    "load_clean_arenas_dataset",

    # Training utilities
    "train_node_classification",
    "train_link_prediction",
]