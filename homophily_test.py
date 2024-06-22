import argparse
import os
from pathlib import Path

from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as f
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from datasets import Dataset

from utils.homophily_metrics import random_disassortative_splits, classifier_based_performance_metric, similarity, \
    adjusted_homo, class_controlled_feature_homophily, attribute_homophily, density_aware_homophily, localsim_cos_homophily, localsim_euc_homophily, two_hop_homophily,\
    label_informativeness, node_homophily, our_measure, edge_homophily, generalized_edge_homophily, neighborhood_homophily, cross_class_neighbor_similarity
from utils.util_funcs import row_normalized_adjacency, sys_normalized_adjacency, normalize_tensor, \
    sparse_mx_to_torch_sparse_tensor

ifsum = 1
num_exp = 10

ACMGCN_FEATURES_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/acmgcn_features/'
Path(ACMGCN_FEATURES_PATH).mkdir(parents=True, exist_ok=True)

BASE_CLASSIFIERS = ['kernel_reg0', 'kernel_reg1', 'gnb']


# SMALL_DATASETS = ['cornell', 'wisconsin', 'texas', 'film', 'chameleon', 'squirrel', 'cora', 'citeseer', 'pubmed']
# LARGE_DATASETS = ['deezer-europe', 'Penn94', 'arxiv-year', "genius", "twitch-gamer", 'pokec', 'snap-patents']
# DATASETS = SMALL_DATASETS + LARGE_DATASETS

SMALL_DATASETS = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers',
    'squirrel-filtered', 'chameleon-filtered', 'actor', 'texas-4-classes', 'cornell', 'wisconsin',
    'cora','corafull','citeseer','pubmed',
    'amazon-photo','amazon-computer','coauthor-cs','coauthor-physics','wikics','blog-catalog',
    'twitch-DE','twitch-ENGB','twitch-ES','twitch-FR','twitch-PTBR','twitch-RU','twitch-TW']
LARGE_DATASETS = ['ogbn-arxiv','genius','questions','flickr']
DATASETS = SMALL_DATASETS + LARGE_DATASETS


METRIC_LIST = {
    # Label Homophily
    "node_homo": lambda adj, labels: node_homophily(adj, labels),
    "edge_homo": lambda adj, labels: edge_homophily(adj, labels),
    "class_homo": lambda adj, labels: our_measure(adj, labels),
    "adj_homo": lambda adj, labels: adjusted_homo(adj, labels),
    "den_homo": lambda adj, labels: density_aware_homophily(adj, labels),
    "two_hop_homo": lambda adj, labels: two_hop_homophily(adj, labels),
    "neibh_homo": lambda adj, labels: neighborhood_homophily(adj, labels),
    # Structural Homophily
    "label_info": lambda adj, labels: label_informativeness(adj, labels),
    "ccns": lambda adj,labels: cross_class_neighbor_similarity(adj, labels),
    # Feature Homophily
    "agg_homo_soft": lambda x: np.mean(x),
    "agg_homo_hard": lambda x: np.mean(x),
    "node_hom_generalized": lambda adj, features, labels: generalized_edge_homophily(adj, features, labels),
    "cls_ctrl_feat_homo": lambda adj, features, labels: class_controlled_feature_homophily(adj, features, labels),
    "attr_homo": lambda adj, features, labels: attribute_homophily(adj, features, labels),
    "localsim_cos_homo": lambda adj, features: localsim_cos_homophily(adj, features),
    "localsim_euc_homo": lambda adj, features: localsim_euc_homophily(adj, features),
    # Classifer-based Homophily
    "kernel_reg0_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "kernel_reg1_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "gnb_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "svm_linear_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "svm_rbf_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "svm_poly_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
}




from config import get_args
args = get_args()

dataset_name = args.dataset
device=args.device
homophily_metric = args.homophily_metric
homophily_lvl = -1

dataset = Dataset(name=args.dataset,
                    model_name=args.model,
                    add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                    device=args.device,
                    use_sgc_features=args.use_sgc_features,
                    use_identity_features=args.use_identity_features,
                    use_adjacency_features=args.use_adjacency_features,
                    do_not_use_original_features=args.do_not_use_original_features,
                    topk=args.topk,
                    toprank=args.toprank,
                    syn_num_node=args.syn_num_node,
                    syn_num_class=args.syn_num_class,
                    syn_num_degree=args.syn_num_degree,
                    syn_feat_dim=args.syn_feat_dim,
                    syn_label_homophily=args.syn_h_l,
                    syn_structural_homophily=args.syn_h_s,
                    syn_feature_homomophily=args.syn_h_f,
                    syn_train_ratio=args.syn_train_ratio,
                    syn_test_ratio=args.syn_test_ratio,
                    seed=args.seed)

edge_index = torch.stack(dataset.graph.edges())
adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), [dataset.num_node, dataset.num_node])
adj = normalize_tensor(adj.to_dense(),
                        symmetric=args.symmetric).to(device)
features = dataset.node_features
labels = dataset.labels
adj = adj.to_sparse().to(device)
features = features.to(device)
labels = labels.to(device)

if homophily_metric=='h_s':
    h_N,h_N_lst = dataset.structural_homophily
    homophily_lvl = h_N
elif homophily_metric=='h_f':
    h_F,_ = dataset.feature_homophily
    homophily_lvl = h_F
elif homophily_metric in ("node_homo", "class_homo", "label_info", "adj_homo", "den_homo", "two_hop_homo","neibh_homo","ccns"):
    homophily_lvl = METRIC_LIST[homophily_metric](adj, labels)
elif homophily_metric == "edge_homo":
    homophily_lvl = METRIC_LIST[homophily_metric](adj, labels)
elif homophily_metric in ("cls_ctrl_feat_homo","attr_homo"):
    homophily_lvl = METRIC_LIST[homophily_metric](adj, features, labels)
elif homophily_metric == "node_hom_generalized":
    homophily_lvl = METRIC_LIST[homophily_metric](adj, features, labels)
elif homophily_metric in ("localsim_cos_homo","localsim_euc_homo"):
    homophily_lvl = METRIC_LIST[homophily_metric](adj, features)
elif homophily_metric in ("agg_homo_soft", "agg_homo_hard"):
    # adj, features, labels = full_load_data_large(dataset_name)
    nnodes = dataset.num_node
    las = np.zeros(10)
    is_hard = 1 if homophily_metric.partition("agg_homo_")[-1] == "hard" else None
    # Compute aggregation Homophily
    num_sample = 10000
    label_onehot = torch.eye(labels.max() + 1)[labels].to(device)
    for i in range(10):
        if nnodes >= num_sample:
            idx_train, _, _ = random_disassortative_splits(labels, labels.max() + 1, num_sample / nnodes)
        else:
            idx_train = None
        las[i] = 2 * similarity(label_onehot, adj, label_onehot, hard=is_hard, LP=1, idx_train=idx_train) - 1
    homophily_lvl = METRIC_LIST[homophily_metric](las)
elif "based_homo" in homophily_metric:
    base_classifier = homophily_metric.partition("_based")[0]
    # adj, features, labels = full_load_data_large(dataset_name)
    homophily_lvl,_,acc_X,acc_G = METRIC_LIST[homophily_metric](features, adj, labels, args.sample_max,
                                                  base_classifier=base_classifier, epochs=100)

print(f"The Homophily level of given dataset {dataset_name} is {homophily_lvl} using metric {homophily_metric}")

# if args.save_result:
#     result = {'homophily_lvl':float(homophily_lvl)}
#     if "based_homo" in homophily_metric:
#         result.update({"acc_X":acc_X,"acc_G":acc_G})
#     result.update(vars(args))
#     result['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
#     df = pd.DataFrame(columns=result.keys())
#     df = df.append(result, ignore_index=True)
#     if os.path.exists(args.save_result_path):
#         df.to_csv(args.save_result_path,mode='a',header=False) 
#     else:
#         df.to_csv(args.save_result_path,mode='w',header=True) 