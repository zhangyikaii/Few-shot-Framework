from models.few_shot.base import FewShotModel

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.metrics import pairwise_distances
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
from typing import Callable

from models.backbone.classification_heads import ClassificationHead, MetaOptNetHead_SVM_CS
from models.utils import create_query_label

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

        loss = loss_fn(logits, y)

        if prefix == 'train_':
            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss
    return core

class MetaOptNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.cls_head = ClassificationHead(base_learner='SVM-CS').cuda()

    def _forward(self, support, query, prefix):
        cur_way = eval(f'self.args.{prefix}way')
        cur_shot = eval(f'self.args.{prefix}shot')

        y_support = create_query_label(cur_way, cur_shot).repeat(self.args.meta_batch_size, 1).to(torch.device('cuda'))

        logit_query = self.cls_head(query, support, y_support, cur_way, cur_shot)

        
        return logits, None
