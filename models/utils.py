import os, shutil
import torch

def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass

def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


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
    parser.add_argument('--num_tasks', type=int, default=1000)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--drop_lr_every', type=int, default=40)
    parser.add_argument('--model_class', type=str, default='ProtoNet', 
                        choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT', 'FEATSTAR', 'SemiFEAT', 'SemiProtoFEAT']) # None for MatchNet or ProtoNet   

    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the  
    
    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1) # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
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
    
    return parser

import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


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
    
    args.parm_str = f'{args.dataset}_{args.way}-way_{args.shot}-shot__{args.eval_way}-eval-way_{args.eval_shot}-eval-shot__' \
            f'{args.query}-query_{args.eval_query}_eval_query'
    return args
    