import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU

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
            for _ in range(num_layers - 2):
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
        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_, branch=None):
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

def fast_contextual_regularization(embeddings, attributes, margin=1.0, metric='euclidean', max_pairs=10000):
    N = embeddings.shape[0]
    device = embeddings.device
    if N < 2 or attributes is None:
        return embeddings.sum() * 0.0
    context_ids = torch.tensor([hash(str(attr)) for attr in attributes], device=device)
    idx_i = torch.randint(0, N, (max_pairs,), device=device)
    idx_j = torch.randint(0, N, (max_pairs,), device=device)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if len(idx_i) == 0 or len(idx_j) == 0:
        return embeddings.sum() * 0.0
    if metric == 'cosine':
        emb_i = F.normalize(embeddings[idx_i], p=2, dim=1)
        emb_j = F.normalize(embeddings[idx_j], p=2, dim=1)
        dist = 1 - (emb_i * emb_j).sum(dim=1)
    else:
        dist = torch.norm(embeddings[idx_i] - embeddings[idx_j], dim=1)
    same_ctx = (context_ids[idx_i] == context_ids[idx_j])
    if same_ctx.any():
        same_ctx_loss = dist[same_ctx].mean()
    else:
        same_ctx_loss = embeddings.sum() * 0.0
    if (~same_ctx).any():
        diff_ctx_loss = F.relu(margin - dist[~same_ctx]).mean()
    else:
        diff_ctx_loss = embeddings.sum() * 0.0
    return same_ctx_loss + diff_ctx_loss

def contrastive_loss(x, x_aug, temperature=0.2, sym=True):
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
        contrastive_loss_value = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        contrastive_loss_value = - torch.log(loss_1).mean()
    return contrastive_loss_value

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse):
        super(GCL, self).__init__()
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse)

    def forward(self, x, Adj_, branch=None):
        z, embedding = self.encoder(x, Adj_, branch)
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, attributes=None, temperature=0.2, sym=True, context_weight=0.1, margin=1.0, context_mode=False, distance_metric='euclidean', context_pair_samples=10000):
        contrastive_loss_value = contrastive_loss(x, x_aug, temperature=temperature, sym=sym)
        context_loss = x.sum() * 0.0  # fallback zero with grad
        if context_mode and attributes is not None:
            ctx_loss = fast_contextual_regularization(
                x, attributes, margin=margin, metric=distance_metric, max_pairs=context_pair_samples
            )
            context_loss = ctx_loss
        total_loss = contrastive_loss_value + context_weight * context_loss
        return total_loss, contrastive_loss_value, context_loss

    @staticmethod
    def calc_context_loss(x, x_aug, attributes=None, margin=1.0, context_mode=True, distance_metric='euclidean', context_pair_samples=10000):
        context_loss = x.sum() * 0.0  # fallback zero with grad
        if context_mode and attributes is not None:
            ctx_loss = fast_contextual_regularization(
                x, attributes, margin=margin, metric=distance_metric, max_pairs=context_pair_samples
            )
            context_loss = ctx_loss
        return context_loss