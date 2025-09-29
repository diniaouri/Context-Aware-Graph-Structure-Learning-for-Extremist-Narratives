import argparse
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from preprocessing import (
    SchemaA1Dataset,
    FullFrenchTweetDataset,
    OldSchemaA1Dataset,
    ARENASFrenchAnnotator1Dataset,
    ARENASFrenchAnnotator2Dataset,
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
    ARENASGermanAnnotator1Dataset,
    ARENASGermanAnnotator2Dataset,
    ARENASCypriotAnnotator1Dataset,
    ARENASCypriotAnnotator2Dataset,
    ARENASSloveneAnnotator1Dataset,
    ARENASSloveneAnnotator2Dataset,
)
from model import GCN, GCL
from graph_learners import FGP_learner, ATT_learner, GNN_learner, MLP_learner
import dgl
import os
import random
from utils import save_loss_plot, ExperimentParameters, accuracy, get_feat_mask, symmetrize, normalize, split_batch, torch_sparse_eye, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse

from sklearn.neighbors import kneighbors_graph

EOS = 1e-10

def sanitize_filename_part(s):
    # Replace any characters that could break filenames, especially slashes
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")

def build_run_suffix(args, max_parts=4):
    """
    Create a unique suffix for the run based on key arguments.
    Limits the length to avoid OS filename errors.
    max_parts: How many argument groups to include (exp_nb, context, context_columns, plus one more group).
    """
    suffix = []
    suffix.append(f"exp{args.exp_nb}")
    # Always include context_mode if set
    if getattr(args, "context_mode", False):
        suffix.append("context")
    # Always include context_columns if set
    if getattr(args, "context_columns", None):
        # SAFETY: Replace any unsafe characters in column names
        safe_columns = [sanitize_filename_part(col) for col in args.context_columns]
        suffix.append("ctxcols_" + "_".join(sorted(safe_columns)))
    # Only include up to max_parts total groups (after exp_nb/context/context_columns)
    included = 3 # already included exp_nb, context_mode, context_columns
    for arg in vars(args):
        if arg not in ["exp_nb", "context_mode", "context_columns", "embeddings_path"] and getattr(args, arg) is not None:
            if included < max_parts:
                suffix.append(f"{sanitize_filename_part(arg)}_{sanitize_filename_part(getattr(args, arg))}")
                included += 1
    return "__".join(suffix)

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
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        num_classes = max(predictions.max(), labels.max()) + 1
        correct = predictions == labels
        accuracies = torch.tensor([
            correct[labels == i].float().mean() if (labels == i).any() else 0.0
            for i in range(num_classes)
        ])
        with open("./accuracies.txt", "a") as f: 
            line = "/".join(map(str, accuracies.cpu().numpy().flatten()))
            f.write(line + '\n')
        return accuracies

    def per_class_acc_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        accu = self.per_class_accuracy(logp[mask], labels[mask])
        return accu

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args, context_dataset=None):
        if getattr(args, "maskfeat_rate_anchor", None):
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        if getattr(args, "maskfeat_rate_learner", None):
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not getattr(args, "sparse", False):
            learned_adj = symmetrize(learned_adj)
            learned_adj = learned_adj.detach()
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        if context_dataset is not None and getattr(args, "context_mode", False):
            def get_context_attributes(indices):
                # Use all columns specified from terminal, including ones with spaces
                return context_dataset.get_context_attributes(indices, columns=args.context_columns)
        else:
            def get_context_attributes(indices):
                return None

        if getattr(args, "contrast_batch_size", None):
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, args.contrast_batch_size)
            total_loss = 0
            total_contrast = 0
            total_context = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                attributes = get_context_attributes(batch)
                out = model.calc_loss(
                    z1[batch], z2[batch], attributes=attributes,
                    context_weight=getattr(args, "context_regularization_weight", 1.0),
                    margin=getattr(args, "context_regularization_margin", 1.0),
                    context_mode=getattr(args, "context_mode", False),
                    distance_metric=getattr(args, "context_distance_metric", "euclidean")
                )
                batch_loss, batch_contrast, batch_context = out
                total_loss += batch_loss * weight
                total_contrast += batch_contrast * weight
                total_context += batch_context * weight
        else:
            attributes = get_context_attributes(list(range(features.shape[0])))
            total_loss, total_contrast, total_context = model.calc_loss(
                z1, z2, attributes=attributes,
                context_weight=getattr(args, "context_regularization_weight", 1.0),
                margin=getattr(args, "context_regularization_margin", 1.0),
                context_mode=getattr(args, "context_mode", False),
                distance_metric=getattr(args, "context_distance_metric", "euclidean")
            )

        return total_loss, total_contrast, total_context, learned_adj

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
            train_mask = torch.tensor(train_mask).cuda()        
            val_mask = torch.tensor(val_mask).cuda()
            test_mask = torch.tensor(test_mask).cuda()
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
        # Only include the first 4 key argument groups (exp_nb, context, context_columns, one more)
        run_suffix = build_run_suffix(args, max_parts=4)
        context_dataset = None

        # ======= DATASET SELECTION, NEW NUMBERING =======
        if args.exp_nb == 1:
            dataset = OldSchemaA1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
        elif args.exp_nb == 2:
            dataset = SchemaA1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 3:
            dataset = FullFrenchTweetDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 4:
            dataset = ARENASFrenchAnnotator1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 5:
            dataset = ARENASFrenchAnnotator2Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 6:
            dataset = ToxigenDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 7:
            dataset = LGBTEnDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 8:
            dataset = MigrantsEnDataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 9:
            dataset = ARENASGermanAnnotator1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 10:
            dataset = ARENASGermanAnnotator2Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 11:
            dataset = ARENASCypriotAnnotator1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 12:
            dataset = ARENASCypriotAnnotator2Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 13:
            dataset = ARENASSloveneAnnotator1Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        elif args.exp_nb == 14:
            dataset = ARENASSloveneAnnotator2Dataset(args.exp_nb, embeddings_path=args.embeddings_path)
            context_dataset = dataset
        else:
            raise ValueError(f"Unknown experiment number: {args.exp_nb}")

        if getattr(args, "gsl_mode", "structure_refinement") == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = dataset.get_dataset()
        elif getattr(args, "gsl_mode", "structure_refinement") == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = dataset.get_dataset()

        for trial in range(getattr(args, "ntrials", 1)):
            self.setup_seed(trial)

            if getattr(args, "gsl_mode", "structure_refinement") == 'structure_inference':
                if getattr(args, "sparse", False):
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif getattr(args, "gsl_mode", "structure_refinement") == 'structure_refinement':
                if getattr(args, "sparse", False):
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw, 'sym', getattr(args, "sparse", False))

            if getattr(args, "sparse", False):
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            type_learner = getattr(args, "type_learner", None)
            if type_learner == 'fgp':
                graph_learner = FGP_learner(
                    features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner)
            elif type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                            args.activation_learner, anchor_adj)
            else:
                raise ValueError(f"Unknown type_learner: {type_learner}. Please check your configuration or command-line arguments. Must be one of: 'fgp', 'mlp', 'att', 'gnn'.")

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
                train_mask = torch.tensor(train_mask).cuda()
                val_mask = torch.tensor(val_mask).cuda()
                test_mask = torch.tensor(test_mask).cuda()
                features = features.cuda()
                labels = torch.tensor(labels).cuda()
                if not getattr(args, "sparse", False):
                    anchor_adj = anchor_adj.cuda()

            loss_list = []

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                total_loss, contrast_loss, context_loss, Adj = self.loss_gcl(
                    model, graph_learner, features, anchor_adj, args, context_dataset=context_dataset
                )

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                total_loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - getattr(args, "tau", 0)) and (getattr(args, "c", 0) == 0 or epoch % getattr(args, "c", 1) == 0):
                    if getattr(args, "sparse", False):
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(
                            Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                            + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(
                            anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                print(f"Epoch {epoch:05d} | Contrastive: {contrast_loss:.4f} | Context: {context_loss:.4f} | Total: {total_loss:.4f}")
                loss_list.append(total_loss.item())

                if epoch % 200 == 0:
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj

                    if getattr(args, "sparse", False):
                        f_adj.edata['w'] = f_adj.edata['w'].detach()
                    else:
                        f_adj = f_adj.detach()
                    os.makedirs("./adjacency_matrices", exist_ok=True)
                    adj_file = f'./adjacency_matrices/adjacency_learned_epoch_{epoch}__{run_suffix}.pkl'
                    with open(adj_file, 'wb') as file:
                        pickle.dump(f_adj, file)

            # --- Save embeddings and labels after training ---
            model.eval()
            with torch.no_grad():
                embeddings, _ = model(features, anchor_adj)
                embeddings_np = embeddings.cpu().numpy()
                emb_file = f'./embeddings/embeddings__{run_suffix}.npy'
                np.save(emb_file, embeddings_np)
            # --- End of embedding saving ---

            # --- Compute and save adjacency from final embeddings ---
            # This ensures adjacency reflects the final embeddings (and thus context)
            # You can choose n_neighbors and metric to suit your application
            adj_final = kneighbors_graph(embeddings_np, n_neighbors=10, metric="cosine", mode='connectivity').toarray()
            os.makedirs("./adjacency_matrices", exist_ok=True)
            adj_final_file = f'./adjacency_matrices/adjacency_final__{run_suffix}.pkl'
            with open(adj_final_file, 'wb') as file:
                pickle.dump(adj_final, file)
            # --- End adjacency saving ---

            save_loss_plot(loss_list, args)

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_nb', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--context_mode', action='store_true', help='Use contextual regularization loss')
    parser.add_argument('--context_columns', nargs='+', default=None,
                        help='List of context columns to use for regularization (can be any columns in your dataset, e.g. Topic "Intolerance" "Hostility to out-group (e.g. verbal attacks, belittlement, instillment of fear, incitement to violence)" "Irony/Humor")')
    parser.add_argument('--embeddings_path', type=str, default=None, help='Custom embeddings file path')
    parser.add_argument('--context_distance_metric', type=str, default='euclidean',
        choices=['euclidean', 'cosine'],
        help='Distance metric for context regularization loss: "euclidean" or "cosine"')
    parser.add_argument('--context_regularization_margin', type=float, default=1.0,
        help='Margin for context regularization loss (default: 1.0)')
    parser.add_argument('--context_regularization_weight', type=float, default=1.0,
        help='Weight for context regularization loss (default: 1.0)')
    return parser.parse_args()

if __name__ == '__main__':
    cl_args = parse_cli()
    # Set the GPU device before anything else that uses CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(cl_args.gpu)
    experiment_params = ExperimentParameters(cl_args.exp_nb)
    experiment_params.context_mode = cl_args.context_mode
    experiment_params.context_columns = cl_args.context_columns
    experiment_params.embeddings_path = cl_args.embeddings_path
    experiment_params.context_distance_metric = cl_args.context_distance_metric
    experiment_params.context_regularization_margin = cl_args.context_regularization_margin
    experiment_params.context_regularization_weight = cl_args.context_regularization_weight

    print(experiment_params.type_learner)
    experiment = Experiment()
    experiment.train(experiment_params)