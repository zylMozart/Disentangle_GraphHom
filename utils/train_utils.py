import os
import yaml
import numpy as np
import torch
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sla
from datetime import datetime
import pandas as pd

def normalize_tensor(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 1)
    if symmetric == 0:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)
        return mx
    
    
def spectral_radius_sp_matrix(edge_index,values,num_nodes):
    '''
    Compute spectral radius for a sparse matrix using scipy
    Input
        edge_index: edge set from src to dst (2,num_edges)
        values: weight for the edge (num_edges)
    Output
        Spectral radius of the matrix
    '''
    adj = coo_matrix((values, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    eigenvalues, eigenvectors = sla.eigs(adj, k=1)
    return np.abs(eigenvalues[0])

def sym_matrix(A,device):
    n = A.shape[0]
    indices = torch.triu_indices(n, n).to(device)
    matrix = torch.zeros(n, n).to(device)
    matrix[indices[0], indices[1]] = A[indices[0], indices[1]]
    matrix = matrix.t()
    matrix[indices[0], indices[1]] = A[indices[0], indices[1]]
    return matrix

class Logger:
    def __init__(self, args, metric, num_data_splits):
        self.all_args = args
        self.eval_class_wise = args.eval_class_wise
        self.save_result = args.save_result
        self.save_dir = args.save_dir
        self.verbose = args.verbose
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.test_metric_cls_lst = []
        self.num_runs = args.num_runs
        self.num_data_splits = num_data_splits
        self.cur_run = None
        self.cur_data_split = None
        # early_stopping
        self.early_stopping = args.early_stopping
        self.best_val = -1
        self.best_step = -1

        # print(f'Results will be saved to {self.save_dir}.')
        # with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
        #     yaml.safe_dump(vars(args), file, sort_keys=False)

    def early_stopping_update(self,val_acc,cur_step):
        if val_acc>self.best_val:
            self.best_val = val_acc
            self.best_step = cur_step
            return True
        else:
            if cur_step-self.best_step>self.early_stopping:
                return False
            else:
                return True

    def start_run(self, run, data_split):
        self.cur_run = run
        self.cur_data_split = data_split
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.test_metric_cls_lst.append([])
        self.best_steps.append(None)

        if self.num_data_splits == 1:
            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Starting run {run}/{self.num_runs} (using data split {data_split}/{self.num_data_splits})...')

    def update_metrics(self, metrics, test_metric_cls, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.test_metric_cls_lst[-1] = test_metric_cls
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.4f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.4f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.4f}')

    def finish_run(self):
        print(f'Finished run {self.cur_run}. '
              f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
              f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
              f'(step {self.best_steps[-1]}).\n')
        if self.cur_run==self.num_runs:
            self.save_metrics()

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        test_metric_mean = np.mean(self.test_metrics).item()
        test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan
        result = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            "test_metric_cls_lst": self.test_metric_cls_lst,
            'best steps': self.best_steps
        }
        print(f'Finished {result["num runs"]} runs.')
        print(f'Val {self.metric} mean: {result[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {result[f"val {self.metric} std"]:.4f}')
        print(f'Test {self.metric} mean: {result[f"test {self.metric} mean"]:.4f}')
        print(f'Test {self.metric} std: {result[f"test {self.metric} std"]:.4f}')
        if self.save_result:
            result.update(vars(self.all_args))
            result['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            df = pd.DataFrame(columns=result.keys())
            df = df.append(result, ignore_index=True)
            if os.path.exists(self.save_dir):
                df.to_csv(self.save_dir,mode='a',header=False) 
            else:
                df.to_csv(self.save_dir,mode='w',header=True) 
        # with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
        #     yaml.safe_dump(metrics, file, sort_keys=False)

    # def print_metrics_summary(self):
    #     # with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
    #     #     metrics = yaml.safe_load(file)

    #     print(f'Finished {metrics["num runs"]} runs.')
    #     print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
    #     print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
    #     print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
    #     print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
