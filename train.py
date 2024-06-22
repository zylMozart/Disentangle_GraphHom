from config import get_args
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from model import Model
from datasets import Dataset
from utils.train_utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup

def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])

    if loss.isnan():
        raise Exception("Nan in loss, please fix it!")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False, eval_class_wise=0):
    model.eval()

    with autocast(enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features)

    metrics = dataset.compute_metrics(logits,eval_class_wise)

    return metrics


def main():
    args = get_args()

    torch.manual_seed(args.seed)

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
    # args.est_h_l=float(dataset.label_homophily)
    if args.eval_class_wise==1:
        args.est_h_s=list(dataset.structural_homophily[1].numpy())
    if args.toprank>=0:
        args.est_h_f=float(dataset.feature_homophily[0])
    logger = Logger(args, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                      num_layers=args.num_layers,
                      input_dim=dataset.num_node_features,
                      hidden_dim=args.hidden_dim,
                      output_dim=dataset.num_targets,
                      hidden_dim_multiplier=args.hidden_dim_multiplier,
                      num_heads=args.num_heads,
                      normalization=args.normalization,
                      residual=args.residual,
                      dropout=args.dropout)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics, test_metric_cls = evaluate(model=model, dataset=dataset, amp=args.amp, eval_class_wise=args.eval_class_wise)
                logger.update_metrics(metrics=metrics, test_metric_cls=test_metric_cls, step=step)
                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})
                if not logger.early_stopping_update(metrics[f'val {dataset.metric}'],step):
                    break

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    # logger.print_metrics_summary()


if __name__ == '__main__':
    main()
