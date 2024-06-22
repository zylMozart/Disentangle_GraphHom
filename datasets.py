import os
import numpy as np
import torch
from torch.nn import functional as F
import dgl
from dgl import ops
from sklearn.metrics import roc_auc_score
from utils.util_funcs import sym_matrix,spectral_radius_sp_matrix
import time
from  torch.distributions import multivariate_normal
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce,spmm
from tqdm import tqdm

class Dataset:
    def __init__(self, name, model_name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False,topk=1,toprank=-1,
                 syn_num_node=None,syn_num_class=None,syn_num_degree=None,syn_feat_dim=None,
                 syn_label_homophily=None,syn_structural_homophily=None,syn_feature_homomophily=None,
                 syn_train_ratio=None,syn_test_ratio=None,seed=None):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        # print('Preparing data...')
        if name!='syn':
            data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))
        else:
            data = self.load_syn_data(model_name,syn_num_node,syn_num_class,syn_num_degree,syn_feat_dim,
                 syn_label_homophily,syn_structural_homophily,syn_feature_homomophily,
                 syn_train_ratio,syn_test_ratio,seed,device)
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])
        if edges.shape[0]==2:
            edges = edges.t()
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)

        if 'directed' not in name:
            graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        node_features = self.augment_node_features(graph=graph,
                                                   node_features=node_features,
                                                   use_sgc_features=use_sgc_features,
                                                   use_identity_features=use_identity_features,
                                                   use_adjacency_features=use_adjacency_features,
                                                   do_not_use_original_features=do_not_use_original_features)
        # if topk!=1:
        #     node_features = self.filter_node_features(graph=graph,
        #                                             node_features=node_features,
        #                                             topk=topk,toprank=toprank)
            
        if toprank>=0:
            node_features = node_features[:,[toprank]]

        node_features = self.transform_node_features(node_features)

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets
        self.num_node = len(labels)
        self.num_edge = edges.shape[0]
        self.num_class = len(labels.unique())

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits, eval_class_wise):
        test_metric_cls=[]
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            if eval_class_wise==1:
                pred_success = (preds[self.test_idx] == self.labels[self.test_idx])
                for c in range(self.num_class):
                    test_acc_cls = pred_success[self.labels[self.test_idx]==c].float().mean().item()
                    test_metric_cls.append(test_acc_cls)
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics,test_metric_cls

    @staticmethod
    def augment_node_features(graph, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = graph.num_nodes()
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix().to_dense()
            node_features = torch.cat([node_features, adj_matrix], axis=1)

        return node_features

    @staticmethod
    def transform_node_features(node_features):
        means = node_features.mean(dim=0)
        means = means.repeat(node_features.shape[0],1)
        stds = node_features.std(dim=0)
        stds[torch.where(stds==0)]=1
        stds = stds.repeat(node_features.shape[0],1)
        return (node_features-means)/stds

    @staticmethod
    def filter_node_features(graph, node_features, topk=0.5, toprank=1):
        feat_norm = (node_features-node_features.mean(dim=0))/node_features.std(dim=0)
        dirchlet = ops.u_sub_v(graph, feat_norm, feat_norm)
        dirchlet = (0.5*dirchlet*dirchlet).sum(dim=0)
        selected = torch.topk(toprank*dirchlet, round(len(dirchlet)*topk))[1]
        new_features = node_features[:,selected]
        # print(node_features.shape)
        # print(new_features.shape)
        return new_features

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features

    def load_syn_data(self, model_name,syn_num_node,syn_num_class,syn_num_degree,syn_feat_dim,
                 syn_label_homophily,syn_structural_homophily,syn_feature_homomophily,syn_train_ratio,syn_test_ratio, seed, device):
        # if model_name=='ResNet':
            num_splits = 1
            edge_index, feat, label = random_graph_with_feature(syn_num_node,syn_num_class,syn_num_degree,syn_feat_dim,
                    syn_label_homophily,syn_structural_homophily,syn_feature_homomophily, seed, device)
            masks = torch.stack([torch.randperm(len(label)) for i in range(num_splits)])
            train_masks = (masks>=(len(label)*syn_train_ratio))
            test_masks = (masks<len(label)*syn_test_ratio)
            val_masks = torch.logical_not(torch.logical_or(train_masks,test_masks))
            data = {'node_features':feat.to('cpu'),'node_labels':label.to('cpu'),'edges':edge_index.to('cpu'),
                    'train_masks':train_masks.to('cpu'),'val_masks':val_masks.to('cpu'),'test_masks':test_masks.to('cpu')}
            # np.save('data/syn/temp',data)
            return data
        # else:
        #     data = np.load('data/syn/temp.npy',allow_pickle=True)
        #     return data.item()

    # @property
    # def inter_class_distance(self):
    #     features = self.node_features.to('cpu')
    #     labels = self.labels.long().to('cpu')
    #     mean_features = torch.zeros((self.num_class,features.shape[1]))
    #     for c in range(self.num_class):
    #         mean_features[c] = features[labels==c].mean(dim=0)
    #     res=0
    #     for i in range(self.num_class):
    #         for j in range(i):
    #             res+=(mean_features[i]-mean_features[j]).square().mean()
    #     res=res/(self.num_class*(self.num_class-1)/2)
    #     return float(res)

    # @property
    # def intra_class_distance(self):
    #     features = self.node_features.to('cpu')
    #     labels = self.labels.long().to('cpu')
    #     var_features = torch.zeros((self.num_class,features.shape[1]))
    #     for c in range(self.num_class):
    #         var_features[c] = features[labels==c].var(dim=0)
    #     var_features=var_features.nan_to_num(0)
    #     return float(var_features.mean())

    @property
    def label_homophily(self, type='edge'):
        edges = self.graph.edges()
        labels = self.labels.to('cpu')
        if type=='edge':
            edges = remove_self_loops(torch.stack(edges))[0].long()
            src_label = labels[edges[0]]
            targ_label = labels[edges[1]]
            labeled_edges = (src_label >= 0) * (targ_label >= 0)
            return torch.mean((src_label[labeled_edges] == targ_label[labeled_edges]).float())
        elif type=='node':
            edge_index = remove_self_loops(torch.stack(edges))[0].long()
            hs = torch.zeros(len(labels))
            degs = torch.bincount(edge_index[0,:]).float()
            matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
            hs = hs.scatter_add(0, edge_index[0,:], matches) / degs
            return hs[degs != 0].mean()

    @property
    def structural_homophily(self):
        edges = self.graph.edges()
        edges = remove_self_loops(torch.stack(edges))[0].long().to('cpu')
        A, value = coalesce(edges, torch.ones(edges.shape[1]), self.num_node, self.num_node)
        labels = self.labels.long().to('cpu')
        Y = F.one_hot(labels)
        dist = spmm(A, value, self.num_node, self.num_node, Y)
        # if (dist.sum(dim=1)==0).sum()>0:
        #     raise Exception("Jaylen: Zero degree nodes exists, please fix it!")
        dist = F.normalize(dist,p=1,dim=1)
        def get_max_std(c):
            return np.sqrt((1-1/c)/c)
        h_N = []
        for c in range(self.num_class):
            c_dist = dist[labels==c]
            if c_dist.shape[0]==1:
                continue
            else:
                std_list = c_dist.std(dim=0)
                std_max = get_max_std(c_dist.shape[1])
                h_N_item = (1-std_list/std_max).mean()
                h_N.append(h_N_item)
        h_N = torch.stack(h_N)
        num_node_class = Y.sum(dim=0)/Y.sum()
        if self.name=='texas-4-classes' : num_node_class = num_node_class[[0,2,3,4]]
        # h_N_cls_weighted = (h_N*num_node_class).sum()
        return std_list.mean(), h_N
        # return h_N_cls_weighted, h_N, std_list.mean()
    
    @property
    def feature_homophily(self):
        edges = self.graph.edges()
        edges = remove_self_loops(torch.stack(edges))[0].long().to('cpu')
        A = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (self.num_node, self.num_node)).to(self.device)
        I = torch.sparse_coo_tensor(torch.arange(self.num_node).repeat(2,1), torch.ones(self.num_node), (self.num_node, self.num_node)).to(self.device)
        X = self.node_features.to(self.device)
        Y = 1.0*F.one_hot(self.labels.long()).to(self.device)

        spectral_radius = spectral_radius_sp_matrix(edges,torch.ones(edges.shape[1]),self.num_node)

        h_F_lst = 0.01*torch.arange(-100,101) # Range From [-1,1] with step=0.01
        v_lst = []
        for h_F in tqdm(h_F_lst):
            w = h_F/spectral_radius
            X0 = torch.sparse.mm(I-w*A,X)
            X0_cls = ((X0.t()@Y)/Y.sum(dim=0).repeat(self.num_node_features,1))
            X0_cls = (X0_cls@Y.t()).t()
            v = torch.pow(X0_cls-X0,2).sum(dim=0)
            v_lst.append(v)
        v_lst = torch.stack(v_lst)

        h_F_feat = []
        for f in range(self.num_node_features):
            h_F = round(float(h_F_lst[int(torch.argmin(v_lst[:,f]))]),2)
            # if (v_lst[:,f]==v_lst[:,f].min()).sum()>2:
            #     print(f"more than 2 minimum! {(v_lst[:,f]==v_lst[:,f].min()).sum()}")
            h_F_feat.append(h_F)
        h_F_graph = h_F_lst[int(torch.argmin(v_lst.sum(dim=1)))]
        # print("h_F of {:>20}: {:.2f}|{:.2f}| ".format(self.name,h_F_graph,np.mean(h_F_feat)),end='')
        # print(h_F_feat[:10])

        # return h_F_graph,np.mean(h_F_feat)
        return np.mean(h_F_feat),spectral_radius

def random_graph_with_feature(
        num_node,num_class,node_degree,feat_dim,
        label_homophily,structural_homophily,feature_homophily,seed,device
        ):
    """Generate random graphs with features sampled from labels and neighbors
    
    Parameters:
    num_node -- int. Number of total node number
    num_class -- int. Number of class number
    node_degree -- string of "lowest highest". the node degrees follow a uniform distribution
    feat_dim -- feature dimension
    label_homophily -- float, ranging from [0,1]
    structural_homophily -- float, ranging from [0,1]
    feature_homophily -- float, ranging from (-1,1)
    """
    start = time.time()
    np.random.seed(seed)
    # Prepare data format
    num_node = int(num_node)
    num_class = int(num_class)
    feat_dim = int(feat_dim)
    node_degree = [int(d) for d in node_degree.split(' ')]
    D = torch.randint(node_degree[0], node_degree[1], (num_node,)).to(device)
    D = torch.diag(D)

    # Labels Nx1 & one-hot labels NxC
    Y = torch.randint(0, num_class, (num_node,)).to(device)
    Z = F.one_hot(Y)*1.0
    # Sample matrix CxC
    S = torch.ones(num_class,num_class).to(device)*(1-label_homophily)/(num_class-1)
    S.fill_diagonal_(label_homophily)

    Nei_dist = Z@S + torch.normal(mean=0, 
                                  std=np.sqrt(np.power(1-structural_homophily,2)/(num_class-1)),
                                #   std=30*(1-structural_homophily),
                                  size=(num_node,num_class)).to(device)

    # bernoulli sampling
    A_p = num_class/num_node*D.pow(0.5)@Nei_dist@Z.t()@D.pow(0.5)
    A_p[A_p>1]=1
    A_p[A_p<0]=0
    A = torch.bernoulli(A_p)
    A = sym_matrix(A,device)

    # Feature
    edge_index = A.nonzero()
    spectral_radius = spectral_radius_sp_matrix(edge_index.to('cpu').t(),torch.ones(edge_index.shape[0]),num_node)
    X0 = torch.zeros(num_node, feat_dim).to(device)
    for d in range(feat_dim):
        C_mean = torch.rand(num_class).to(device)
        C_vars = torch.rand(num_class).to(device)
        X_mean = Z@C_mean
        X_vars = Z@C_vars
        X0[:,d] = multivariate_normal.MultivariateNormal(X_mean,torch.diag(X_vars)).sample()
    nei_info = torch.matrix_power(torch.eye(num_node).to(device)-feature_homophily/spectral_radius*A,-1)
    X = nei_info@X0
    end = time.time()
    # print(end-start)
    return edge_index.to('cpu'),X.to('cpu'),Y.to('cpu')



datasets = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
    'squirrel-filtered', 'chameleon-filtered', 'actor', 'texas-4-classes', 'cornell', 'wisconsin',
    'cora','corafull','citeseer','pubmed','flickr',
    'amazon-photo','amazon-computer','coauthor-cs','coauthor-physics','wikics','blog-catalog',
    'ogbn-arxiv',
    'genius','twitch-DE','twitch-ENGB','twitch-ES','twitch-FR','twitch-PTBR','twitch-RU','twitch-TW']

if __name__=='__main__':
    from config import get_args
    dataset_name = "cora"
    args = get_args()
    dataset = Dataset(name=dataset_name,
                    model_name=args.model,
                    device='cuda:0',
                    seed=args.seed)
    h_L = dataset.label_homophily
    h_N,h_N_lst = dataset.structural_homophily
    h_F,_ = dataset.feature_homophily
    print(f"Dataset: {dataset}")
    print(f"Label homophily: {h_L}")
    print(f"Structural homophily: {h_N}")
    print(f"Feature homophily: {h_F}")
    pass
