import math
import random
import time

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB
from torch_scatter import scatter_add

from utils.util_funcs import random_disassortative_splits, accuracy
import dgl
from dgl import ops


pi = math.pi
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
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

def cross_class_neighbor_similarity(A, labels):
    """Cross-Class Neighborhood Similarity (CCNS) https://arxiv.org/pdf/2106.06134.
        Since the author only proposes a higher inter-class NS and a low intra-class NS
        improves GNNs without defining a metric, we implement this idea as the ratio
    """
    ### Using for-loop to prevent the 
    num_nodes = labels.shape[0]
    num_class = labels.unique().shape[0]
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    edge_index = remove_self_loops(edge_idx)[0]
    dist = torch.zeros((num_nodes,num_class))
    for i in range(num_class):
        dist_1c = torch.zeros(num_nodes).to(device)
        dist_1c = dist_1c.scatter_add_(0,src_node,1.0*(labels[targ_node]==i).long())
        dist[:,i] = dist_1c
    dist = F.normalize(dist,p=1,dim=1)
    all_ns = (torch.pow(dist.sum(dim=0),2)-torch.pow(dist,2).sum(dim=0)).sum()
    cls_ns = 0
    for i in range(num_class):
        c_dist = dist[labels==i]
        cls_ns += (torch.pow(c_dist.sum(dim=0),2)-torch.pow(c_dist,2).sum(dim=0)).sum()
        pass
    result = ((all_ns-cls_ns)/(num_class*(num_class-1)))/(cls_ns/num_class)
    return result

def neighborhood_homophily(A, labels):
    """ Default 2-hop
    """
    num_nodes = labels.shape[0]
    num_class = labels.unique().shape[0]
    A = A@A
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    edge_index = remove_self_loops(edge_idx)[0]
    dist = torch.zeros((num_nodes,num_class))
    for i in range(num_class):
        dist_1c = torch.zeros(num_nodes).to(device)
        dist_1c = dist_1c.scatter_add_(0,src_node,1.0*(labels[targ_node]==i).long())
        dist[:,i] = dist_1c
    hom_node = dist.max(dim=1)[0]/dist.sum(dim=1)
    hom_node = hom_node.nan_to_num(0)
    return hom_node.mean()

def two_hop_homophily(A, labels):
    """ average of homophily for each node
    """
    A = A@A
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    edge_index = remove_self_loops(edge_idx)[0]
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)

def localsim_euc_homophily(A, features):
    # features_std = (features.std(dim=0).nan_to_num(1))
    # features_std[features_std==0] = 1
    # features = (features-features.mean(dim=0))/features_std
    num_node, num_feat = features.shape[0], features.shape[1]
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    graph = dgl.graph((src_node,targ_node))
    out = ops.u_sub_v(graph, features, features)
    # out = out.norm(p=2,dim=1)
    sim = -out.norm(p=2,dim=1)
    degs = torch.bincount(src_node).float()
    degs[degs==0] = 1 # To prevent zero divide
    sim_node = torch.zeros(num_node).to(device)
    sim_node = sim_node.scatter_add_(0,src_node,sim)
    sim_node = sim_node/degs
    return sim_node.mean()

def localsim_cos_homophily(A, features):
    num_node, num_feat = features.shape[0], features.shape[1]
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    graph = dgl.graph((src_node,targ_node))
    out = ops.u_mul_v(graph, features, features)
    out = out.sum(dim=1)
    out_norm = ops.u_mul_v(graph, features.norm(dim=1), features.norm(dim=1))
    sim = out/out_norm
    sim[sim.isnan()] = 0
    degs = torch.bincount(src_node).float()
    degs[degs==0] = 1 # To prevent zero divide
    sim_node = torch.zeros(num_node).to(device)
    sim_node = sim_node.scatter_add_(0,src_node,sim)
    sim_node = sim_node/degs
    return sim_node.mean()


def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j
     of edges incident to class i nodes
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = edge_idx
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze().to(device)
    c = label.max()+1
    H = torch.zeros((c,c)).to(device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx, device=device).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    return H

def density_aware_homophily(A, labels):
    label = labels.squeeze()
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_index = torch.stack([src_node,targ_node])
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    c = nonzero_label.max() + 1
    counts = F.one_hot(nonzero_label.long(), c).sum(0).float()
    complete_graph_edges = counts.view(-1,1).mm(counts.view(1, -1))
    complete_graph_edges.diag() / 2 + counts
    try:
        h = H[counts!=0,:][:,counts!=0]/complete_graph_edges[counts!=0,:][:,counts!=0]
    except RuntimeError as e:
        raise RuntimeError('Missing labels')
    h_homo = h.diag()
    h_hete = (h.triu(1) + h.tril(-1)).max(0).values
    # ret = max(h_hete, h_homo) * (h_homo - h_hete) / density
    # return 1 / (1 + torch.exp(- ret))
    return (h_homo - h_hete).min()/2+0.5

def attribute_homophily(A, features, labels):
    assert(torch.logical_not(labels.unique()>=0).sum()==0) # All the labels must start from 0
    assert(len(labels.unique())==labels.max()+1)
    num_class = labels.unique().shape[0]
    num_node, num_feat = features.shape[0], features.shape[1]
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    graph = dgl.graph((src_node,targ_node))
    out = ops.u_mul_v(graph, features, features)
    out = out.transpose(1,0)
    feat_agg = torch.zeros(num_feat, num_node).to(device)
    feat_agg = feat_agg.scatter_add_(1,src_node.repeat(num_feat,1),out)
    feat_agg = feat_agg.transpose(1,0)
    degs = torch.bincount(src_node).float()
    degs[degs==0] = 1 # To prevent zero divide
    feat_agg = feat_agg/degs.unsqueeze(1).repeat(1,num_feat)
    feat_sum = features.sum(dim=0)
    feat_sum[feat_sum==0]=1
    feat_agg = feat_agg.sum(dim=0)
    hom = feat_agg/feat_sum
    return hom.mean()

def class_controlled_feature_homophily(A, features, labels):
    assert(torch.logical_not(labels.unique()>=0).sum()==0) # All the labels must start from 0
    assert(len(labels.unique())==labels.max()+1)
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    num_class = labels.unique().shape[0]
    num_node, num_feat = features.shape[0], features.shape[1]
    feature_cls = torch.zeros(num_class,num_feat).to(device)
    for c in range(num_class):
        feature_cls[c] = features[labels==c].mean(dim=0)
    cls_ctrl_feat = features - feature_cls[labels]
    base_hom = (cls_ctrl_feat-cls_ctrl_feat.mean(dim=0).repeat(num_node,1))
    base_hom = 2*base_hom*base_hom
    base_hom = base_hom.sum(dim=1)
    node_pair_distance = (cls_ctrl_feat[src_node]-cls_ctrl_feat[targ_node])*(cls_ctrl_feat[src_node]-cls_ctrl_feat[targ_node])
    node_pair_distance = node_pair_distance.sum(dim=1)
    node_pair_CFH = base_hom[src_node] - node_pair_distance[src_node]
    node_level_CFH = torch.zeros(num_node).to(device)
    node_level_CFH = node_level_CFH.scatter_add_(0,src_node,node_pair_CFH)
    degs = torch.bincount(src_node).float()
    degs[degs==0] = 1 # To prevent zero divide
    node_level_CFH = node_level_CFH
    graph_level_CFH = node_level_CFH.mean()
    mean_b = base_hom.mean()
    mean_d = torch.zeros(num_node).to(device)
    mean_d = mean_d.scatter_add_(0,src_node,node_pair_distance).mean()
    norm_graph_level_CFH = graph_level_CFH/max(mean_b,mean_d)
    return norm_graph_level_CFH

def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.coalesce().indices()[0, :], A.coalesce().indices()[1, :]  # A.nonzero()
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
    edge_idx = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes).to(device)
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


def our_measure(A, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_index = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
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


def class_distribution(A, labels):
    edge_index = A.coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    deg = torch.bincount(src_node).float()

    # remove self-loop
    deg = deg - 1
    edge_index = remove_self_loops(A.coalesce().indices())[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]

    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    num_class = int(labels.max().cpu() + 1)
    p_bar = torch.zeros(num_class).to(device)
    pc = torch.zeros((num_class, num_class)).to(device)
    for i in range(num_class):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])

        for j in range(num_class):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc


def adjusted_homo(A, label):
    p, p_bar, pc = class_distribution(A, label)
    edge_homo = edge_homophily(A, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))

    return adj_homo


def label_informativeness(A, label):
    p, p_bar, pc = class_distribution(A, label)
    LI = 2 - torch.sum(pc * torch.log(pc)) / torch.sum(p_bar * torch.log(p_bar))
    return LI


def generalized_edge_homophily(adj, features, label, sample_max=75000, iteration=10):
    nedges = adj.coalesce().indices()[0, :].shape[0]
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
        weight_matrix = (torch.zeros(adj.clone().detach().size(0), labels.clone().detach().max() + 1)).to(device)
    else:
        labels = torch.argmax(label, 1)[idx_train]
        label = label[idx_train, :]
        weight_matrix = (torch.zeros(torch.sum(idx_train.int()), labels.clone().detach().max() + 1)).to(device)
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
        elif 'svm' in base_classifier:
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
        else:
            raise Exception("Classifier Not Found!!!")
        
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

    return g_aware_good_p, time.time() - t_time, float(X_results.mean()), float(G_results.mean())
    # return g_aware_good_p, time.time() - t_time
