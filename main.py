import argparse
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from preprocessing import SchemaA1Dataset, FullFrenchTweetDataset, OldSchemaA1Dataset, SelectedDataset
from model import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, GNN_learner, MLP_learner
import dgl
import os
import random
from utils import save_loss_plot, ExperimentParameters, accuracy, get_feat_mask, symmetrize, normalize, split_batch, torch_sparse_eye, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse

EOS = 1e-10


class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def per_class_accuracy(self, predictions, labels):
        """
        Calculate per-class accuracy using vectorized operations

        Args:
            predictions: tensor of model predictions (N, num_classes) or (N,) if already argmaxed
            labels: tensor of true labels (N,)

        Returns:
            Tensor of accuracies for each class
        """
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        num_classes = max(predictions.max(), labels.max()) + 1
        # Create mask for each class
        correct = predictions == labels
        accuracies = torch.tensor([
            correct[labels == i].float().mean() if (labels == i).any() else 0.0
            for i in range(num_classes)
        ])
        with open("/home/cytech/Work/Master/Projet de recherche/Application/accuracies.txt", "a") as f:
            line = "/".join(map(str, accuracies.cpu().numpy().flatten()))
            f.write(line + '\n')
        return accuracies

    def per_class_acc_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        accu = self.per_class_accuracy(logp[mask], labels[mask])
        return accu

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = learned_adj.detach()
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj

    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(
                    model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu = self.loss_cls(
            best_model, test_mask, features, labels)
        per_class_test_acc = self.per_class_acc_cls(
            best_model, test_mask, features, labels)
        return best_val, test_accu, best_model, per_class_test_acc

    def train(self, args):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if args.exp_nb == 1:
            dataset = OldSchemaA1Dataset(args.exp_nb)
        elif args.exp_nb == 2 or args.exp_nb == 3:
            dataset = SchemaA1Dataset(args.exp_nb)
        elif args.exp_nb == 4:
            dataset = FullFrenchTweetDataset(args.exp_nb)
        elif args.exp_nb == 5:
            dataset = SelectedDataset(args.exp_nb)
        elif args.exp_nb == 6:
            dataset = SchemaA1Dataset(args.exp_nb)

        if args.gsl_mode == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = dataset.get_dataset()
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = dataset.get_dataset()
        # print("---------------------Feature Matrix-------------------------------")
        # print(features)
        # print(features.shape)
        # print(f"Number of features : {nfeats}")
        # print("------------------------------Labels----------------------------")
        # print(labels)
        # print(labels.shape)
        # print(labels.unique())
        # print(f"Number of classes : {nclasses}")
        # print("---------------------------Masks----------------------------------")
        # print("Training Mask :\n")
        # print(train_mask)
        # print(train_mask.shape)
        # unique, counts = train_mask.unique(return_counts=True)
        # occurrences = dict(zip(unique.tolist(), counts.tolist()))
        # print(occurrences)
        # print("Val Mask :\n")
        # print(val_mask)
        # print(val_mask.shape)
        # unique, counts = val_mask.unique(return_counts=True)
        # occurrences = dict(zip(unique.tolist(), counts.tolist()))
        # print(occurrences)
        # print("Test Mask :\n")
        # print(test_mask)
        # print(test_mask.shape)
        # unique, counts = test_mask.unique(return_counts=True)
        # occurrences = dict(zip(unique.tolist(), counts.tolist()))
        # print(occurrences)
        # if args.gsl_mode == 'structure_refinement':
        #     print("-----------------------------Adj Matrix----------------------")
        #     print(adj_original)
        # torch.exit(-1)
        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):

            self.setup_seed(trial)

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(
                    features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner, anchor_adj)

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                        emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(
                graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_epoch = 0

            loss_list = []

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                loss, Adj = self.loss_gcl(
                    model, graph_learner, features, anchor_adj, args)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(
                            Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                            + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(
                            anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | CL Loss {:.4f}".format(
                    epoch, loss.item()))
                loss_list.append(loss.item())

                if epoch % 200 == 0:
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj

                    if args.sparse:
                        f_adj.edata['w'] = f_adj.edata['w'].detach()
                    else:
                        f_adj = f_adj.detach()
                    os.makedirs("./adjacency_matrices", exist_ok=True)
                    with open(f'./adjacency_matrices/adjacency_learned_epoch_{epoch}_exp{args.exp_nb}.pkl', 'wb') as file:
                        pickle.dump(f_adj, file)
            save_loss_plot(loss_list, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_nb', type=int)
    cl_args = parser.parse_args()

    experiment_params = ExperimentParameters(cl_args.exp_nb)
    print(experiment_params.type_learner)
    experiment = Experiment()
    experiment.train(experiment_params)
