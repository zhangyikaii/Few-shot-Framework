from models.few_shot.base import FewShotModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.metrics import pairwise_distances
from models.utils import create_nshot_task_label
from typing import Callable, Tuple

# 可能不能继承 FewShotModel.
class Maml(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

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
        
    def _forward(self, support, query):
        pass
