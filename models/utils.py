import os, shutil
import os.path as osp
import torch
import numpy as np
from torch.utils.data import DataLoader

def mkdir(dirs):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def rmdir(dirs):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    if os.path.exists(dirs):
        shutil.rmtree(dirs)

def get_command_line_parser():
    """解析命令行参数.

    命令行参数说明:
        TODO

    # Arguments
        None
    # Return
        argparse.ArgumentParser(), 还需要 .parse_args() 转.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB', 'OmniglotDataset'])
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
    
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--drop_lr_every', type=int, default=40)
    parser.add_argument('--model_class', type=str, default='ProtoNet', 
                        choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 'SemiFEAT', 'SemiProtoFEAT']) # None for MatchNet or ProtoNet   
    parser.add_argument('--logger_filepath', type=str)

    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the  

    parser.add_argument('--loss_fn', type=str, default='F-cross_entropy',
                        choices=['F-cross_entropy', 'nn-cross_entropy'])
    parser.add_argument('--meta_batch_size', default=32, type=int)
    
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    
    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005) # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    parser.add_argument('--test_model_filepath', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    return parser

import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def set_logger(args, logger_name):
    import logging
    logging.basicConfig(
        filename=osp.abspath(osp.dirname(osp.dirname(__file__))) + f'{args.logger_filepath}/{args.params_str}.log',
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    
    return logging.getLogger(logger_name)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('Using gpu:', x)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    return torch.device('cuda')

def preprocess_args(args):
    """根据命令行参数附加处理.

    添加参数:
        TODO
    
    # Argument
        args: parser.parse_args()
    # Return
        处理后的args
    """
    # TODO: setup_dirs 应该在这里根据传入参数建.

    set_gpu(args.gpu)

    # 添加由数据集决定的参数:
    if args.dataset == 'OmniglotDataset':
        args.num_input_channels = 1
    elif args.dataset == 'MiniImageNet':
        args.num_input_channels = 3
    
    from time import gmtime, strftime
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    args.params_str = f'{args.model_class}_{args.dataset}_{args.backbone_class}-backbone_{args.distance}' \
            f'_{args.way}-way_{args.shot}-shot__{args.eval_way}-eval-way_{args.eval_shot}-eval-shot__' \
            f'{args.query}-query_{args.eval_query}-eval-query_{time_str}'
    args.train_mode = True if args.test_model_filepath is None else False
    args.model_filepath = f'/mnt/data3/lus/zhangyk/models/{args.model_class}/{args.params_str}.pth' \
        if args.test_model_filepath is None else args.test_model_filepath
    # 在此之后 test_model_filepath 没有用了.

    return args

def create_nshot_task_label(way: int, query: int) -> torch.Tensor:
    """Creates an shot-shot task label.

    Label has the structure:
        [0]*query + [1]*query + ... + [way-1]*query

    # Arguments
        way: Number of classes in the shot-shot classification task
        query: Number of query samples for each class in the shot-shot classification task

    # Returns
        y: Label vector for shot-shot task of shape [query * way, ]
    """

    y = torch.arange(0, way, 1 / query).long() # 很精妙, 注意强转成long了.
    # 返回从 0 ~ way - 1 (label), 每个元素有 query 个(query samples).
    return y