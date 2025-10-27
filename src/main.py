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
import pandas as pd
from utils import save_loss_plot, ExperimentParameters, accuracy, get_feat_mask, symmetrize, normalize, split_batch, torch_sparse_eye, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse

from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

def sanitize_filename_part(s):
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")

def build_run_suffix(args, max_parts=4):
    suffix = []
    suffix.append(f"exp{args.exp_nb}")
    if getattr(args, "context_mode", False):
        suffix.append("context")
    if getattr(args, "context_columns", None):
        safe_columns = [sanitize_filename_part(col) for col in args.context_columns]
        suffix.append("ctxcols_" + "_".join(sorted(safe_columns)))
    included = 3
    for arg in vars(args):
        if arg not in ["exp_nb", "context_mode", "context_columns", "embeddings_path"] and getattr(args, arg) is not None:
            if included < max_parts:
                suffix.append(f"{sanitize_filename_part(arg)}_{sanitize_filename_part(getattr(args, arg))}")
                included += 1
    return "__".join(suffix)

def make_context_adjacency(context_data, context_columns):
    N = context_data.shape[0]
    adj_context = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            any_shared = any(context_data.iloc[i][col] == context_data.iloc[j][col] for col in context_columns)
            adj_context[i, j] = 1.0 if any_shared else 0.0
    np.fill_diagonal(adj_context, 0)
    return adj_context

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

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args, context_dataset=None, dynamic_context_weight=None):
        maskfeat_rate_anchor = args.maskfeat_rate_anchor
        maskfeat_rate_learner = args.maskfeat_rate_learner
        mask_v1, _ = get_feat_mask(features, maskfeat_rate_anchor)
        features_v1 = features * (1 - mask_v1)
        z1, _ = model(features_v1, anchor_adj, 'anchor')
        mask, _ = get_feat_mask(features, maskfeat_rate_learner)
        features_v2 = features * (1 - mask)
        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = learned_adj.detach()
            learned_adj = normalize(learned_adj, 'sym', args.sparse)
        z2, _ = model(features_v2, learned_adj, 'learner')

        # Context attributes
        def get_context_attributes(indices):
            if context_dataset is None or not args.context_mode:
                #print("MAIN DEBUG: context_dataset is None or context_mode is off")
                return None
            #print("MAIN DEBUG: dataset columns:", list(context_dataset.data.columns))
            #print("MAIN DEBUG: args.context_columns:", args.context_columns)
            try:
                attrs = context_dataset.get_context_attributes(indices, columns=args.context_columns)
                #print("MAIN DEBUG: attributes example:", attrs[:10])
                #print("MAIN DEBUG: unique tuples:", len(set(attrs)), "of", len(attrs))
                return attrs
            except Exception as e:
                #print("MAIN DEBUG: Exception in get_context_attributes:", str(e))
                return None

        attributes = get_context_attributes(list(range(features.shape[0])))

        context_weight = dynamic_context_weight if dynamic_context_weight is not None else args.context_regularization_weight

        if args.context_only:
            total_loss = context_weight * GCL.calc_context_loss(
                z1, z2, attributes=attributes,
                margin=args.context_regularization_margin,
                context_mode=args.context_mode,
                distance_metric=args.context_distance_metric,
                context_pair_samples=args.context_pair_samples
            )
            contrast_loss = torch.tensor(0.0)
            context_loss = total_loss / context_weight if context_weight > 0 else total_loss
        else:
            total_loss, contrast_loss, context_loss = GCL.calc_loss(
                z1, z2, 
                attributes=attributes,
                temperature=args.temperature,
                sym=args.sym,
                context_weight=context_weight,
                margin=args.context_regularization_margin,
                context_mode=args.context_mode,
                distance_metric=args.context_distance_metric,
                context_pair_samples=args.context_pair_samples
            )
        return total_loss, contrast_loss, context_loss, learned_adj

    def train(self, args):
        run_suffix = build_run_suffix(args, max_parts=4)
        context_dataset = None

        # ======= DATASET SELECTION =======
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
            # --- Context-based adjacency matrix ---
            use_context_adj = getattr(args, "use_context_adj", False)
            if use_context_adj and args.context_columns is not None:
                if isinstance(features, np.ndarray):
                    df_context = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(features.shape[1])])
                    context_cols = []
                    for col in args.context_columns:
                        if isinstance(col, int):
                            context_cols.append(f'feat_{col}')
                        else:
                            context_cols.append(str(col))
                    df_context = df_context[context_cols]
                else:
                    df_context = features[args.context_columns]
                adj_context = make_context_adjacency(df_context, df_context.columns)
                anchor_adj_raw = torch.from_numpy(adj_context.astype(np.float32)).float()
            else:
                if getattr(args, "gsl_mode", "structure_inference") == 'structure_inference':
                    if getattr(args, "sparse", False):
                        anchor_adj_raw = torch_sparse_eye(features.shape[0])
                    else:
                        anchor_adj_raw = torch.eye(features.shape[0]).float()
                elif getattr(args, "gsl_mode", "structure_refinement") == 'structure_refinement':
                    if getattr(args, "sparse", False):
                        anchor_adj_raw = adj_original
                    else:
                        if isinstance(adj_original, list):
                            adj_original = np.array(adj_original)
                        if isinstance(adj_original, np.ndarray):
                            if adj_original.ndim == 1 or adj_original.shape[0] == 0:
                                adj_original = np.eye(features.shape[0], dtype=np.float32)
                            elif adj_original.ndim == 0:
                                raise ValueError("adj_original is scalar, not adjacency matrix!")
                            elif adj_original.ndim == 2:
                                adj_original = adj_original.astype(np.float32)
                        print("adj_original shape:", adj_original.shape)
                        anchor_adj_raw = torch.from_numpy(adj_original).float()
            anchor_adj = normalize(anchor_adj_raw, 'sym', getattr(args, "sparse", False))
            if getattr(args, "sparse", False):
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)
            type_learner = args.type_learner
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
                raise ValueError(f"Unknown type_learner: {type_learner}.")
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
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()
            loss_list = []
            contrastive_loss_list = []
            context_loss_list = []

            start_weight = 0.0
            max_weight = args.context_regularization_weight
            rampup_epochs = int(args.epochs * 0.6)

            for epoch in range(1, args.epochs + 1):
                if args.context_mode and not args.context_only:
                    if epoch < rampup_epochs:
                        dynamic_context_weight = start_weight + (max_weight - start_weight) * (epoch / rampup_epochs)
                    else:
                        dynamic_context_weight = max_weight
                elif args.context_only:
                    dynamic_context_weight = max_weight
                else:
                    dynamic_context_weight = 0.0

                model.train()
                graph_learner.train()
                total_loss, contrast_loss, context_loss, Adj = self.loss_gcl(
                    model, graph_learner, features, anchor_adj, args, context_dataset=context_dataset,
                    dynamic_context_weight=dynamic_context_weight
                )
                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                total_loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                loss_list.append(total_loss.item())
                contrastive_loss_list.append(contrast_loss.item())
                context_loss_list.append(context_loss.item())

                print(f"Epoch {epoch:05d} | Contrastive: {contrast_loss:.4f} | Context: {context_loss:.4f} | Total: {total_loss:.4f} | ContextWeight: {dynamic_context_weight:.4f}")

                if epoch % 200 == 0:
                    model.eval()
                    graph_learner.eval()
                    f_adj = Adj
                    if args.sparse:
                        f_adj.edata['w'] = f_adj.edata['w'].detach()
                    else:
                        f_adj = f_adj.detach()
                    os.makedirs("./adjacency_matrices", exist_ok=True)
                    adj_file = f'./adjacency_matrices/adjacency_learned_epoch_{epoch}__{run_suffix}.pkl'
                    with open(adj_file, 'wb') as file:
                        pickle.dump(f_adj, file)

            epochs = np.arange(1, len(loss_list) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, loss_list, label="Total Loss")
            plt.plot(epochs, contrastive_loss_list, label="Contrastive Loss")
            plt.plot(epochs, context_loss_list, label="Context Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Evolution Over Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"losses_separate__{run_suffix}.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, loss_list)
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss")
            plt.title("Total Loss Over Epochs")
            plt.tight_layout()
            plt.savefig(f"loss_total__{run_suffix}.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, contrastive_loss_list)
            plt.xlabel("Epoch")
            plt.ylabel("Contrastive Loss")
            plt.title("Contrastive Loss Over Epochs")
            plt.tight_layout()
            plt.savefig(f"loss_contrastive__{run_suffix}.png")
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(epochs, context_loss_list)
            plt.xlabel("Epoch")
            plt.ylabel("Context Loss")
            plt.title("Context Loss Over Epochs")
            plt.tight_layout()
            plt.savefig(f"loss_context__{run_suffix}.png")
            plt.close()

            np.save(f"losses_total_{run_suffix}.npy", np.array(loss_list))
            np.save(f"losses_contrastive_{run_suffix}.npy", np.array(contrastive_loss_list))
            np.save(f"losses_context_{run_suffix}.npy", np.array(context_loss_list))

            model.eval()
            with torch.no_grad():
                embeddings, _ = model(features, anchor_adj)
                embeddings_np = embeddings.cpu().numpy()
                emb_file = f'./embeddings/embeddings__{run_suffix}.npy'
                np.save(emb_file, embeddings_np)
            adj_final = kneighbors_graph(embeddings_np, n_neighbors=args.n_neighbors, metric="cosine", mode='connectivity').toarray()
            os.makedirs("./adjacency_matrices", exist_ok=True)
            adj_final_file = f'./adjacency_matrices/adjacency_final__{run_suffix}.pkl'
            with open(adj_final_file, 'wb') as file:
                pickle.dump(adj_final, file)
            save_loss_plot(loss_list, args)

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_nb', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--context_mode', action='store_true')
    parser.add_argument('--context_only', action='store_true', help='Use only contextual regularization (no contrastive loss)')
    parser.add_argument('--use_context_adj', action='store_true', help='Use context-based adjacency matrix')
    parser.add_argument('--context_columns', nargs='+', default=None)
    parser.add_argument('--embeddings_path', type=str, default=None)
    parser.add_argument('--context_distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    parser.add_argument('--context_regularization_margin', type=float, default=0.05) #to test
    parser.add_argument('--context_regularization_weight', type=float, default=0.1)#to test
    parser.add_argument('--context_pair_samples', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rep_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropedge_rate', type=float, default=0.2)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--type_learner', type=str, default='fgp', choices=['fgp', 'mlp', 'att', 'gnn'])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--sim_function', type=str, default='cosine')
    parser.add_argument('--activation_learner', type=str, default='relu')
    parser.add_argument('--gsl_mode', type=str, default='structure_refinement', choices=['structure_refinement', 'structure_inference'])
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--w_decay', type=float, default=5e-4)
    parser.add_argument('--sym', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    cl_args = parse_cli()
    if torch.cuda.is_available():
        torch.cuda.set_device(cl_args.gpu)
    experiment_params = ExperimentParameters(cl_args.exp_nb)
    for arg in vars(cl_args):
        setattr(experiment_params, arg, getattr(cl_args, arg))
    print(experiment_params.type_learner)
    experiment = Experiment()
    experiment.train(experiment_params)
