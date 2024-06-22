import torch
import numpy as np
import dgl
import scipy
import csv
import json
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Amazon, Coauthor, AttributedGraphDataset, WikiCS
import torch_geometric.transforms as T
from ogb.nodeproppred import NodePropPredDataset
import argparse

def load_new_dataset(dataset_name,split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10):
    '''Download and preprocess new dataset'''
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # DGL datasets
    if dataset_name in ['cora','corafull','pubmed','citeseer','flickr']:
        node_features, node_labels, edges = get_dgl_dataset(dataset_name)
    # PYG datasets
    elif dataset_name in ['amazon-photo','amazon-computer','coauthor-cs','coauthor-physics',
                          'wikics','blog-catalog','ppi','facebook']:
        node_features, node_labels, edges = get_pyg_dataset(dataset_name)
    # OGB datasets
    elif 'ogbn' in dataset_name:
        node_features, node_labels, edges = get_ogb_dataset(dataset_name)
    elif dataset_name=='genius' or 'fb100' in dataset_name or 'twitch' in dataset_name:
        node_features, node_labels, edges = get_large_nonhom_dataset(dataset_name)
    # Splits
    num_nodes = len(node_labels)
    mask_number = torch.zeros(num_data_splits,num_nodes)
    for i in range(num_data_splits):
        mask_number[i] = torch.randperm(num_nodes)
    train_masks = (mask_number<=(train_prop*num_nodes))
    val_masks = (torch.logical_and(mask_number<=((train_prop+valid_prop)*num_nodes),mask_number>(train_prop*num_nodes)))
    test_masks = (mask_number>((train_prop+valid_prop)*num_nodes))

    # Save as npz
    dataset_name = dataset_name.replace('-','_')
    np.savez(f'data/{dataset_name}.npz',
                node_features=node_features.numpy(),
                node_labels=node_labels.numpy(),
                edges=edges.numpy(),
                train_masks=train_masks.numpy(),
                val_masks=val_masks.numpy(),
                test_masks=test_masks.numpy())
    pass

def get_dgl_dataset(dataset_name):
    maps = {'cora':'CoraGraphDataset',
            'corafull':'CoraFullDataset',
            'citeseer':'CiteseerGraphDataset',
            'pubmed':'PubmedGraphDataset',
            'flickr':'FlickrDataset'}
    dataset = getattr(dgl.data,maps[dataset_name])()
    graph = dataset[0]
    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    edges = graph.edges()
    edges = torch.stack(edges).transpose(1,0)
    return node_features, node_labels, edges

def get_pyg_dataset(dataset_name):
    transform = T.NormalizeFeatures()
    if dataset_name=='amazon-photo':
        dataset = Amazon(root='data/temp', name='Photo', transform=transform)
        dataset = dataset[0]
        return dataset.x, dataset.y, dataset.edge_index
    elif dataset_name=='amazon-computer':
        dataset = Amazon(root='data/temp', name='Computers', transform=transform)
        dataset = dataset[0]
        return dataset.x, dataset.y, dataset.edge_index
    elif dataset_name == 'coauthor-cs':
        dataset = Coauthor(root='data/temp',name='CS', transform=transform)
        return dataset.x, dataset.y, dataset.edge_index
    elif dataset_name == 'coauthor-physics':
        dataset = Coauthor(root='data/temp',name='Physics', transform=transform)
        return dataset.x, dataset.y, dataset.edge_index
    elif dataset_name == 'wikics':
        dataset = WikiCS(root='data/temp')
        dataset = dataset[0]
        return dataset.x, dataset.y, dataset.edge_index
    elif dataset_name == 'blog-catalog':
        dataset = AttributedGraphDataset(root='data/temp', name='BlogCatalog')
        dataset = dataset[0]
        return dataset.x, dataset.y, dataset.edge_index

def get_ogb_dataset(dataset_name):
    dataset = NodePropPredDataset(name=dataset_name, root='data/temp')
    node_features = dataset.graph['node_feat']
    node_labels = dataset.labels
    edges = dataset.graph['edge_index']
    node_features = torch.tensor(node_features)
    node_labels = torch.tensor(node_labels).reshape(-1)
    edges = torch.tensor(edges)
    return node_features, node_labels, edges

def get_large_nonhom_dataset(dataset_name):
    maps={'fb100-amherst41':'Amherst41.mat',
          'fb100-cornell5':'Cornell5.mat',
          'fb100-johnshopkins55':'Johns Hopkins55.mat',
          'fb100-penn94':'Penn94.mat',
          'fb100-reed98':'Reed98.mat',}
    if 'fb100' in dataset_name:
        mat = scipy.io.loadmat(f'data/large_nonhom_dataset/facebook100/{maps[dataset_name]}')
        A = mat['A']
        metadata = mat['local_info']
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        metadata = metadata.astype(np.int)
        label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
        feature_vals = np.hstack(
            (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        features = np.empty((A.shape[0], 0))
        for col in range(feature_vals.shape[1]):
            feat_col = feature_vals[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            features = np.hstack((features, feat_onehot))
        node_feat = torch.tensor(features, dtype=torch.float)
        num_nodes = metadata.shape[0]
        graph = {'edge_index': edge_index,
                        'edge_feat': None,
                        'node_feat': node_feat,
                        'num_nodes': num_nodes}
        label = torch.tensor(label)
        return node_feat, label, edge_index
    elif dataset_name=='genius':
        fulldata = scipy.io.loadmat('data/large_nonhom_dataset/genius.mat')
        edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
        node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
        label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
        return node_feat, label, edge_index
    elif 'twitch' in dataset_name:
        lang = dataset_name.split('-')[-1]
        A, label, features = load_twitch(lang)
        edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
        node_feat = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return node_feat, label, edge_index

def load_twitch(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"data/large_nonhom_dataset/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0] # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label
    return A, label, features


if __name__=='__main__':
    DATASETS = ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions', 
                'squirrel-filtered', 'chameleon-filtered', 'actor', 'texas-4-classes', 'cornell', 
                'wisconsin', 'cora','corafull','citeseer','pubmed','flickr', 'amazon-photo',
                'amazon-computer','coauthor-cs','coauthor-physics','wikics','blog-catalog', 
                'ogbn-arxiv', 'genius','twitch-DE','twitch-ENGB','twitch-ES','twitch-FR',
                'twitch-PTBR','twitch-RU','twitch-TW']
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, default='cora', choices=DATASETS,
                        help=f"The data set name, please select from the following list: \n"
                            f"{DATASETS}")
    parser.add_argument('--split_type', type=str, default='random', 
                        help='Split type for train, valid, and test set for node classification.')
    parser.add_argument('--num_data_splits', type=int, default=10, 
                        help='Number of data splits.')
    parser.add_argument('--train_prop', type=float, default=0.6,
                        help='Proportion of train set.')
    parser.add_argument('--valid_prop', type=float, default=0.2,
                        help='Proportion of validation set.')
    args = parser.parse_args()

    load_new_dataset(dataset_name=args.dataset,split_type=args.split_type, 
                     train_prop=args.train_prop, valid_prop=args.valid_prop, num_data_splits=args.num_data_splits)
    
    # load_new_dataset(dataset_name='corafull',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='citeseer',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='pubmed',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='flickr',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    
    # load_new_dataset(dataset_name='amazon-photo',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='amazon-computer',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='coauthor-cs',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='coauthor-physics',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='wikics',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='blog-catalog',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)

    # load_new_dataset(dataset_name='ogbn-arxiv',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='genius',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)
    # load_new_dataset(dataset_name='fb100-penn94',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10) # With some unlabeled data
    # for sub_name in ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW']:
    #     load_new_dataset(dataset_name=f'twitch-{sub_name}',split_type='random', train_prop=0.6, valid_prop=0.2, num_data_splits=10)