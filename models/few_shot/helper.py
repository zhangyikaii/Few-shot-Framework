import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.few_shot.protonet import ProtoNet
from models.few_shot.maml import MAML

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
        top_para = [v for k,v in model.named_parameters() if ('encoder' not in k and 'args' not in k)]
        # as in the literature, we use ADAM for ConvNet and SGD for other backbones
        if self.args.meta:
            optimizer = optim.Adam(
                [{'params': top_para, 'lr': self.args.lr * self.args.lr_mul}],
                lr=self.args.lr,
                # weight_decay=args.weight_decay, do not use weight_decay here
            )
        else:
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

    def prepare_loss_fn(self):
        if self.args.loss_fn == 'F-cross_entropy':
            return F.cross_entropy
        elif self.args.loss_fn == 'nn-cross_entropy':
            return nn.CrossEntropyLoss()
