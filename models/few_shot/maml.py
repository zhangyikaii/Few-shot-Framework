from models.few_shot.base import FewShotModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.metrics import pairwise_distances
from models.utils import create_nshot_task_label
from typing import Callable, Tuple
from torch.optim import Optimizer



def fit_handle(
    model: nn,
    optimizer: Optimizer,
    loss_fn: Callable
    ):
    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        train: bool = True
        ):
        """
        create_graph: 构造导数图, 允许计算高阶导数的乘积.
        If you have to use this function, make sure to reset the .grad fields of your 
        parameters to None after use to break the cycle and avoid the leak.
        """
        create_graph = (True if order == 2 else False) and train
        
        if train:
            # Zero gradients
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        logits, reg_logits = model(x)
        loss = loss_fn(logits, y)

        if train:
            # Take gradient step
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss
    return core


# 可能不能继承 FewShotModel.
# 没错是的.
class Maml(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    
    def prepare_nshot_task(self, shot: int, way: int, query: int) -> Callable:
        def prepare_meta_batch_(batch):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients
            num_input_channels = 3
            x = x.reshape(self.args.meta_batch_size, shot*way + query*way, num_input_channels, x.shape[-2], x.shape[-1])
            # Move to device
            x = x.double().cuda()
            # Create label
            y = create_nshot_task_label(way, query).cuda().repeat(self.args.meta_batch_size)
            return x, y

        return prepare_meta_batch_

    # 该函数不应作为成员函数.
    def replace_grad(self, parameter_gradients, parameter_name):
        def replace_grad_(module):
            return parameter_gradients[parameter_name]

        return replace_grad_

    def forward(self, x, get_feature=False):
        pass