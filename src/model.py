import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU

# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        if sparse:
            self.layers.append(GCNConv_dgl(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv_dgl(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dgl(hidden_channels, out_channels))
        else:
            self.layers.append(GCNConv_dense(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.layers.append(GCNConv_dense(hidden_channels, hidden_channels))
            self.layers.append(GCNConv_dense(hidden_channels, out_channels))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):

        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        if sparse:
            self.gnn_encoder_layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dgl(hidden_dim, emb_dim))
        else:
            self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self,x, Adj_, branch=None):

        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.proj_head(x)
        return z, x

# --- UPDATED: Accepts metric argument ---
def contextual_regularization(embeddings, attributes, margin=1.0, metric='euclidean'):
    batch_size = embeddings.size(0)
    loss = 0.0
    count = 0
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            same_context = attributes[i] == attributes[j]
            if metric == 'cosine':
                # 1 - cosine similarity is cosine distance
                dist = 1 - F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            else:  # 'euclidean'
                dist = torch.norm(embeddings[i] - embeddings[j])
            if same_context:
                loss += dist
            else:
                loss += torch.relu(margin - dist)
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=embeddings.device)

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, attributes=None, temperature=0.2, sym=True, context_weight=0.1, margin=1.0, context_mode=False, distance_metric='euclidean'):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            contrastive_loss = (loss_0 + loss_1) / 2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            contrastive_loss = - torch.log(loss_1).mean()

        context_loss = torch.tensor(0.0, device=x.device)
        if context_mode and attributes is not None:
            # --- UPDATED: Pass distance_metric argument ---
            context_loss = contextual_regularization(x, attributes, margin=margin, metric=distance_metric)

        total_loss = contrastive_loss + context_weight * context_loss
        return total_loss, contrastive_loss, context_loss