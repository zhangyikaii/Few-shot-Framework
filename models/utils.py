import os, shutil
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from models.few_shot.protonet import ProtoNet

def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.makedirs(dir)
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
    parser.add_argument('--logger_filepath', type=str)

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
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
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
    
    args.params_str = f'{args.model_class}_{args.dataset}_{args.backbone_class}-backbone_{args.distance}_{args.way}-way_{args.shot}-shot__{args.eval_way}-eval-way_{args.eval_shot}-eval-shot__' \
            f'{args.query}-query_{args.eval_query}-eval-query'
    
    return args


class PrepareFunc(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def prepare_model(self):
        # 这里决定了是什么模型.
        # model = ProtoNet(args)
        model = eval(self.args.model_class)(self.args)

        # load pre-trained model (no FC weights)
        # if args.init_weights is not None:
        #     model_dict = model.state_dict()        
        #     pretrained_dict = torch.load(args.init_weights)['params']
        #     if args.backbone_class == 'ConvNet':
        #         pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     print(pretrained_dict.keys())
        #     model_dict.update(pretrained_dict)
        #     model.load_state_dict(model_dict)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        model = model.to(self.device, dtype=torch.double)

        if self.args.multi_gpu:
            model.encoder = nn.DataParallel(model.encoder, dim=0)
            para_model = model.to(self.device, dtype=torch.double)
        else:
            para_model = model.to(self.device, dtype=torch.double)

        return model, para_model

    def prepare_optimizer(self, model):
        top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]       
        # as in the literature, we use ADAM for ConvNet and SGD for other backbones
        if self.args.backbone_class == 'ConvNet':
            optimizer = optim.Adam(
                [{'params': model.encoder.parameters()},
                {'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                lr=self.args.lr,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
        else:
            optimizer = optim.SGD(
                [{'params': model.encoder.parameters()},
                {'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                lr=self.args.lr,
                momentum=self.args.mom,
                nesterov=True,
                weight_decay=self.args.weight_decay
            )        

        # trick
        # 关注step_size等参数.
        if self.args.lr_scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                                optimizer,
                                step_size=int(self.args.step_size),
                                gamma=self.args.gamma
                            )
        elif self.args.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                                optimizer,
                                milestones=[int(_) for _ in self.args.step_size.split(',')],
                                gamma=self.args.gamma,
                            )
        elif self.args.lr_scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                                optimizer,
                                self.args.max_epoch,
                                eta_min=0   # a tuning parameter
                            )
        else:
            raise ValueError('No Such Scheduler')

        return optimizer, lr_scheduler

    def prepare_loss_func(self):
        return nn.CrossEntropyLoss()

        