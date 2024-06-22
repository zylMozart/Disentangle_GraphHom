import math
import random
import time

import numpy as np
import scipy
import torch
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from torch_scatter import scatter_add

# from utils.util_funcs import random_disassortative_splits, accuracy

pi = math.pi
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'
# device = torch.device(device)
device = 'cpu'


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def edge_homophily(edge_index, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = edge_index[0].long(), edge_index[1].long()  # A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = torch.mean(matching.float())
    return edge_hom


def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_index, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_index)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0, :]).float()
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float()
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()


def compact_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx.to(device))[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(edge_index.to(device))
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k, :], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def class_homophily(edge_index, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max() + 1
    H = compact_matrix_edge_idx(edge_index.to(device), label.to(device))
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def class_distribution(edge_index, labels):
    # edge_index = A.coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    deg = src_node.unique(return_counts=True)[1]

    # remove self-loop
    deg = deg - 1
    edge_index = remove_self_loops(edge_index)[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]

    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    p_bar = torch.zeros(labels.max() + 1)
    pc = torch.zeros((labels.max() + 1, labels.max() + 1))
    for i in range(labels.max() + 1):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])

        for j in range(labels.max() + 1):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc


def adjusted_homo(edge_index, label):
    p, p_bar, pc = class_distribution(edge_index, label)
    edge_homo = edge_homophily(edge_index, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))

    return adj_homo


def label_informativeness(edge_index, label):
    label = label.int()
    p, p_bar, pc = class_distribution(edge_index, label)
    LI = 2 - torch.sum(pc * torch.log(pc)) / torch.sum(p_bar * torch.log(p_bar))
    return LI


def generalized_edge_homophily(adj, features, label, nedges, sample_max=75000, iteration=10):
    if nedges < sample_max:
        sim = torch.tensor(cosine_similarity(features.cpu(), features.cpu())).to(device)
        sim[torch.isnan(sim)] = 0
        adj = adj.to_dense()
        adj = adj - torch.diag(torch.diag(adj))
        adj = (adj > 0).float()
        g_edge_homo = torch.sum(sim * adj) / torch.sum(adj)

        return g_edge_homo
    else:
        g_homo = np.zeros(iteration)

        for i in range(iteration):
            sample = torch.tensor(
                random.sample(list(np.arange(adj.coalesce().indices()[0, :].shape[0])), int(sample_max)))
            src_node, targ_node = adj.coalesce().indices()[0, :][sample], adj.coalesce().indices()[1, :][sample]
            sim = torch.sum(features[src_node].cpu() * features[targ_node].cpu(), 1) / \
                  (torch.norm(features[src_node].cpu(), dim=1, p=2) *
                   torch.norm(features[targ_node].cpu(), dim=1, p=2))
            sim[torch.isnan(sim)] = 0
            g_homo[i] = torch.mean(sim)
        return np.mean(g_homo)


def similarity(features, adj, label, hard=None, LP=1, ifsum=1, idx_train=None):
    if str(type(idx_train)) == '<class \'NoneType\'>':
        inner_prod = torch.mm(torch.mm(adj, features), torch.mm(adj, features).transpose(0, 1))
        labels = torch.argmax(label, 1)
        weight_matrix = (torch.zeros(adj.clone().detach().size(0), labels.clone().detach().max() + 1))
    else:
        labels = torch.argmax(label, 1)[idx_train]
        label = label[idx_train, :]
        weight_matrix = (torch.zeros(torch.sum(idx_train.int()), labels.clone().detach().max() + 1))
        inner_prod = torch.mm(torch.spmm(adj, features)[idx_train, :],
                              torch.spmm(adj, features)[idx_train, :].transpose(0, 1))
    for i in range(labels.max() + 1):
        # Think about using torch.sum or torch.mean
        if ifsum == 1:
            weight_matrix[:, i] = torch.sum(inner_prod[:, labels == i], 1)
        else:
            weight_matrix[:, i] = torch.mean(inner_prod[:, labels == i], 1)
    if hard is None:
        if ifsum == 1:
            nnodes = labels.shape[0]
            degs_label = torch.sum(torch.mm(label, label.transpose(0, 1)), 1)
        else:
            nnodes = labels.max() + 1
            degs_label = 1
        if LP == 1:
            # weight mean
            LAF_ratio = (weight_matrix[np.arange(labels.size(0)), labels] / degs_label) / \
                        ((torch.sum(weight_matrix, 1) - weight_matrix[np.arange(labels.size(0)), labels]) / (
                                nnodes - degs_label))
            LAF_ratio[torch.isnan(LAF_ratio)] = 0
            return torch.mean((LAF_ratio >= 1).float())  #
        else:
            return torch.mean(((torch.sum(weight_matrix - weight_matrix * label, 1) <= 0) & (
                    torch.sum(weight_matrix * label, 1) >= 0)).float())
    else:
        if LP == 1:
            return torch.mean(torch.argmax(weight_matrix, 1).eq(labels).float())
        else:
            return torch.mean(((torch.max(weight_matrix - weight_matrix * label, 1)[0] <= 0.) & (
                    torch.sum(weight_matrix * label, 1) >= 0)).float())


def gntk_homophily_(features, adj, sample, n_layers):
    eps = 1e-8
    G_gram = torch.mm(torch.spmm(adj, features)[sample, :],
                      torch.transpose(torch.spmm(adj, features)[sample, :], 0, 1))
    G_norm = torch.sqrt(torch.diag(G_gram)).reshape(-1, 1) * torch.sqrt(torch.diag(G_gram)).reshape(1, -1)
    G_norm = (G_norm > eps) * G_norm + eps * (G_norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(G_gram, G_norm))
        sqrt = torch.sqrt(torch.square(G_norm) - torch.square(G_gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_G = 1 / pi * (G_gram * (pi - arccos) + sqrt)
    else:
        K_G = G_gram

    gram = torch.mm(features[sample, :], torch.transpose(features[sample, :], 0, 1))
    norm = torch.sqrt(torch.diag(gram)).reshape(-1, 1) * torch.sqrt(torch.diag(gram)).reshape(1, -1)
    norm = (norm > eps) * norm + eps * (norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(gram, norm))
        sqrt = torch.sqrt(torch.square(norm) - torch.square(gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_X = 1 / pi * (gram * (pi - arccos) + sqrt)
    else:
        K_X = gram

    return K_G / 2, K_X / 2


def classifier_based_performance_metric(features, adj, labels, sample_max, base_classifier='kernel_reg1', epochs=100):
    nnodes = (labels.shape[0])
    if labels.dim() > 1:
        labels = labels.flatten()
    G_results, X_results, diff_results, G_good_p_results, X_good_p_results = torch.zeros(epochs), torch.zeros(
        epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
    t_time = time.time()
    for j in range(epochs):

        if nnodes <= sample_max:
            sample = np.arange(nnodes)
            label_onehot = torch.eye(labels.max() + 1)[labels].cpu()
            labels_sample = labels.cpu()
        else:
            sample, _, _ = random_disassortative_splits(labels, labels.max() + 1, sample_max / nnodes)
            label_onehot = torch.eye(labels.max() + 1)[labels][sample, :].cpu()
            labels_sample = labels.cpu()[sample]

        idx_train, idx_val, idx_test = random_disassortative_splits(labels_sample, labels_sample.max() + 1)
        idx_val = idx_val + idx_test
        # Kernel Regression based p-values
        if base_classifier in {'kernel_reg0', 'kernel_reg1'}:
            nlayers = 0 if base_classifier == 'kernel_reg0' else 1
            K_graph, K = gntk_homophily_(features, adj, sample, nlayers)
            K_graph_train_train, K_train_train = K_graph[idx_train, :][:, idx_train], K[idx_train, :][:, idx_train]
            K_graph_val_train, K_val_train = K_graph[idx_val, :][:, idx_train], K[idx_val, :][:, idx_train]
            Kreg_G, Kreg_X = K_graph_val_train.cpu() @ (
                    torch.tensor(np.linalg.pinv(K_graph_train_train.cpu().numpy())) @ label_onehot.cpu()[
                idx_train]), K_val_train.cpu() @ (
                                     torch.tensor(np.linalg.pinv(K_train_train.cpu().numpy())) @ label_onehot.cpu()[
                                 idx_train])
            diff_results[j] = (accuracy(labels_sample[idx_val], Kreg_G) > accuracy(labels_sample[idx_val], Kreg_X))
            G_results[j] = accuracy(labels_sample[idx_val],
                                    Kreg_G)
            X_results[j] = accuracy(labels_sample[idx_val],
                                    Kreg_X)
        elif base_classifier == 'gnb':
            #  Gaussian Naive Bayes model
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()

            X_gnb, G_gnb = GaussianNB(), GaussianNB()
            X_gnb.fit(X[idx_train], labels_sample[idx_train])
            G_gnb.fit(X_agg[idx_train], labels_sample[idx_train])

            X_pred = torch.tensor(X_gnb.predict(X[idx_val]))
            G_pred = torch.tensor(G_gnb.predict(X_agg[idx_val]))

            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val]).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(G_pred.eq(labels_sample[idx_val]).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[idx_val]).float())
        else:
            #  SVM based p-values
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()
            if base_classifier == 'svm_rbf':
                G_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_poly':
                G_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_linear':
                G_svm = svm.SVC(kernel='linear').fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='linear').fit(X[idx_train], labels_sample[idx_train])

            G_pred = torch.tensor(G_svm.predict(X_agg[idx_val]))
            X_pred = torch.tensor(X_svm.predict(X[idx_val]))
            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val]).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(G_pred.eq(labels_sample[
                                                    idx_val]).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[
                                                    idx_val]).float())

    if scipy.__version__ == '1.4.1':
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False,
                                                       nan_policy='propagate')
    else:
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False, nan_policy='propagate')

    if torch.mean(diff_results) <= 0.5:
        g_aware_good_p = g_aware_good_p / 2

    else:
        g_aware_good_p = 1 - g_aware_good_p / 2

    return g_aware_good_p, time.time() - t_time
