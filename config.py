import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # basic arguments
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_result', type=int, default=1, help='If save result')
    parser.add_argument('--save_dir', type=str, default='experiments/temp.csv', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='cora',
        choices=['syn',
                'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                'squirrel-filtered', 'chameleon-filtered', 'actor', 'texas-4-classes', 'cornell', 'wisconsin',
                'cora','corafull','citeseer','pubmed','flickr',
                'amazon-photo','amazon-computer','coauthor-cs','coauthor-physics','wikics','blog-catalog',
                'ogbn-arxiv',
                'genius','twitch-DE','twitch-ENGB','twitch-ES','twitch-FR','twitch-PTBR','twitch-RU','twitch-TW'
                ])
    
    # homophily measurement
    parser.add_argument('--symmetric', type=float, default=0,
                        help='1 for symmetric renormalized adj, 0 for random walk renormalized adj')
    parser.add_argument('--sample_max', type=float, default=500, help='maxinum number of samples used in gntk')
    parser.add_argument('--base_classifier', type=str, default='kernel_reg1', 
                        help='The classifier used for performance metric(kernel_reg1, kernel_reg0, svm_linear, svm_rbf, svm_poly, gnb)')
    parser.add_argument('--homophily_metric', default='agg_homo_soft', 
                        help="The metric to measure homophily, please select from the following list: \n"
                            "[ \n"
                            "  node_homo (node homophily), \n"
                            "  edge_homo (edge homophily), \n"
                            "  class_homo (class homophily), \n"
                            "  node_hom_generalized (generalized node homophily), \n"
                            "  agg_homo_soft (aggreation homophily with soft LAS), \n"
                            "  agg_homo_hard (aggreation homophily with hard LAS), \n"
                            "  adj_homo (adjusted homophily), \n"
                            "  label_info (label informativeness), \n"
                            "  kernel_reg0_based_homo (kernel based homophily with reg0), \n"
                            "  kernel_reg1_based_homo (kernel based homophily with reg1), \n"
                            "  gnb_based_homo (gnd-based homophily) \n"
                            "  svm_based_homo (svm-based homophily) \n"
                            "  cls_ctrl_feat_homo (class-controlled feature homophily) \n"
                            "  attr_homo (attribute homophily) \n"
                            "  den_homo (density-aware homophily) \n"
                            "  localsim_cos_homo (Local Similarity homophily) \n"
                            "  localsim_euc_homo (Local Similarity homophily) \n"
                            "  two_hop_homo (2-hop Neighbor Class Similarity) \n"
                            "  neibh_homo (2-hop Neighbor Class Similarity) \n"
                            "  ccns (Cross-Class Neighbor Similarity) \n"
                            "]")
    
    # synthetic dataset setting
    parser.add_argument('--syn_num_node', type=int, default=1000, help='Number of nodes in synthetic datasets.')
    parser.add_argument('--syn_num_class', type=int, default=3, help='Number of classes in synthetic datasets. The datasets are class-balanced')
    parser.add_argument('--syn_num_degree', type=str, default='5 6', help='Minimum and maximum node degree in synthetic datasets. Node degrees follow a uniform distribution.')
    parser.add_argument('--syn_feat_dim', type=int, default=10, help='Number of feature dimensions in synthetic datasets.')
    parser.add_argument('--syn_h_l', type=float, default=0.7, help='Label homophily control in synthetic datasets')
    parser.add_argument('--syn_h_s', type=float, default=0.8, help='Structural homophily control in synthetic datasets')
    parser.add_argument('--syn_h_f', type=float, default=0.5, help='Feature homophily control in synthetic datasets')
    parser.add_argument('--syn_train_ratio', type=float, default=0.5)
    parser.add_argument('--syn_test_ratio', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--est_h_l', type=float, default=0)
    parser.add_argument('--est_h_s', type=float, default=0)
    parser.add_argument('--est_h_f', type=float, default=0)

    # h_F h_S experiment
    parser.add_argument('--topk', type=float, default=1)
    parser.add_argument('--toprank', type=int, default=-1, help='which feature to be selected, >=0 activate')
    parser.add_argument('--eval_class_wise', type=int, default=0, help='Class-wised evaluation, 0 or 1')

    # model architecture
    parser.add_argument('--model', type=str, default='GCN',
                        choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
    parser.add_argument('--residual', type=str, default='residual', choices=['None', 'residual'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--early_stopping', type=int, default=40)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    # node feature augmentation
    parser.add_argument('--use_sgc_features', default=False, action='store_true')
    parser.add_argument('--use_identity_features', default=False, action='store_true')
    parser.add_argument('--use_adjacency_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_original_features', default=False, action='store_true')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args