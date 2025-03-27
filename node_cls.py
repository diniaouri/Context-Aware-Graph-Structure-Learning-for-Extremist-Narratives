import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn import GraphConv, SAGEConv
import pickle
from preprocessing import SchemaA1Dataset
import matplotlib.pyplot as plt
# -------------------------------
# 1. Prepare the Data
# -------------------------------

num_nodes = 100
np.random.seed(42)
with open('./adjacency_matrices/adjacency_learned_epoch_1000_exp3.pkl', 'rb') as f:
    adj_matrix = pickle.load(f)

dataset = SchemaA1Dataset(experiment_nb=3)
# Load node features from dataframe (assuming CSV for example)
node_features = dataset.data
node_features = node_features.dropna(
    subset=["Text"])

node_features["label_str"] = node_features["In-Group"] + \
    "+" + node_features["Out-group"]
node_features = node_features.reset_index(drop=True)


value_counts = node_features['label_str'].value_counts()

# Plot the distribution
plt.figure(figsize=(8, 5))
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Unique IN/OUT group pairs')
plt.xticks(rotation=0)
plt.show()

# One hot encoding
unique_labels = node_features['label_str'].unique()
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
node_features['label'] = node_features['label_str'].map(label_to_num)
nb_classes = len(node_features["label"].unique())
print(f"Nb classes : {nb_classes}")

tweet_embeddings = dataset.embeddings

# Verify alignment between matrix and features
assert adj_matrix.shape[0] == len(
    node_features), "Mismatch between matrix size and feature count"

adj = adj_matrix.numpy()


# -------------------------------
# 2. Create the DGL Graph and Assign Data
# -------------------------------

# Convert the adjacency matrix to an edge list.
src, dst = np.nonzero(adj)
# Create the DGL graph.
graph = dgl.graph((src, dst))
graph = dgl.to_simple(graph)  # remove duplicate edges if any

# Prepare node labels.
labels_tensor = torch.tensor(
    node_features['label'].values, dtype=torch.long)

# Assign node features and labels to the graph.
graph.ndata['feat'] = tweet_embeddings
graph.ndata['label'] = labels_tensor

# Create train and test masks (here we simply use an 80/20 split)
n_nodes = graph.num_nodes()
all_indices = np.arange(n_nodes)
np.random.shuffle(all_indices)
train_size = int(0.4 * n_nodes)
train_idx = all_indices[:train_size]
test_idx = all_indices[train_size:]

train_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

graph.ndata['train_mask'] = train_mask
graph.ndata['test_mask'] = test_mask

# -------------------------------
# 3. Build a GNN Model
# -------------------------------


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(in_feats, hidden_feats)
        self.layer2 = GraphConv(hidden_feats, num_classes)

    def forward(self, graph, features):
        # First graph convolution layer with non-linear activation.
        x = self.layer1(graph, features)
        x = F.relu(x)
        # Second graph convolution layer outputs the logits.
        x = self.layer2(graph, x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, aggregator_type='mean', dropout=0.5):
        super(GraphSAGE, self).__init__()
        # First GraphSAGE layer.
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type)
        # Second GraphSAGE layer outputs logits for each class.
        self.sage2 = SAGEConv(hidden_feats, num_classes, aggregator_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features):
        # Apply first layer and then a ReLU activation.
        h = self.sage1(graph, features)
        h = F.relu(h)
        h = self.dropout(h)
        # Second layer to produce class logits.
        h = self.sage2(graph, h)
        return h


# Hyperparameters
in_feats = tweet_embeddings.shape[1]
hidden_feats = 32
num_classes = len(unique_labels)  # Number of unique combined labels.
learning_rate = 0.01
num_epochs = 1000

# Create the model, loss function, and optimizer.
model = GraphSAGE(in_feats=in_feats, hidden_feats=hidden_feats,
                  num_classes=num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 4. Training Loop
# -------------------------------

for epoch in range(num_epochs):
    model.train()
    # Forward pass: compute logits
    logits = model(graph, graph.ndata['feat'])
    # Compute loss only on the training nodes
    loss = loss_fn(logits[graph.ndata['train_mask']],
                   graph.ndata['label'][graph.ndata['train_mask']])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # Evaluate on training and test sets
        model.eval()
        with torch.no_grad():
            logits = model(graph, graph.ndata['feat'])
            pred = logits.argmax(dim=1)
            train_acc = (pred[graph.ndata['train_mask']] ==
                         graph.ndata['label'][graph.ndata['train_mask']]).float().mean()
            test_acc = (pred[graph.ndata['test_mask']] == graph.ndata['label']
                        [graph.ndata['test_mask']]).float().mean()
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Train Acc {train_acc.item():.4f} | Test Acc {test_acc.item():.4f}")

# -------------------------------
# 5. Final Evaluation
# -------------------------------

model.eval()
with torch.no_grad():
    logits = model(graph, graph.ndata['feat'])
    pred_class = logits.argmax(dim=1)

print("\nFinal node classification results:")
print(pred_class)
