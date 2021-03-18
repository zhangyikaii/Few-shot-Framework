from models.few_shot.base import FewShotModel
from models.backbone.backbone_plus import init_layer

from models.utils import create_onehot_query_label

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

import numpy as np
from typing import Callable, Tuple

from torch.autograd import Variable

# train/val的逻辑, 与模型方法相关.
# 写一个两层的函数, model之类先传
def fit_handle(
    model: nn,
    optimizer: Optimizer,
    scaler: GradScaler,
    loss_fn: Callable
    ):
    # 返回core函数:
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        prefix: str = 'train_'
        ):
        if prefix == 'train_':
            # Zero gradients
            model.train()
        else:
            model.eval()

        logits, reg_logits = model(x, prefix)
        y = y.double() # mse 特性?
        loss = loss_fn(logits, y)

        if prefix == 'train_':
            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss
    return core

class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
        self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

class RelationModule(nn.Module):
    def __init__(self, input_size, hidden_size):        
        super(RelationModule, self).__init__()

        padding = 1 if ( input_size[1] < 10 ) and ( input_size[2] < 10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding)
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding)

        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.fc1 = nn.Linear( input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
        self.fc2 = nn.Linear( hidden_size, 1 )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out

class RelationNet(FewShotModel):
    def __init__(self, args):
        super(RelationNet, self).__init__(args)
        self.relation_module = \
            RelationModule(
                self.encoder.final_feat_dim,
                self.args.relation_hidden_dim
            ).to(torch.device('cuda'))
    
    
    def prepare_kshot_task(self, way: int, query: int, meta_batch_size: int) -> Callable:
        def prepare_kshot_task_onehot_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            x, y = batch
            x = x.to(torch.device('cuda'))
            y = create_onehot_query_label(way, query).to(torch.device('cuda'))
            return x, y
        return prepare_kshot_task_onehot_
    
    def _forward(self, support, query, prefix):
        cur_way   = eval(f'self.args.{prefix}way')
        cur_shot  = eval(f'self.args.{prefix}shot')
        cur_query = eval(f'self.args.{prefix}query')
        z_support = support.view(cur_way, cur_shot, -1)
        z_query   = query.view(cur_way, cur_query, -1)
        
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(cur_way, cur_shot, *self.encoder.final_feat_dim).mean(1) 
        z_query     = z_query.contiguous().view(cur_way * cur_query, *self.encoder.final_feat_dim)

        z_proto_ext = z_proto.unsqueeze(0).repeat(cur_query * cur_way, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(0).repeat(cur_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        extend_final_feat_dim = self.encoder.final_feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, cur_way)

        return relations, None