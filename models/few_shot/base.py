import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple
from models.utils import create_nshot_task_label

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from models.backbone.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from models.backbone.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from models.backbone.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from models.backbone.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def prepare_nshot_task(self, shot: int, way: int, query: int) -> Callable:
        """Typical shot-shot task preprocessing.

        # Arguments
            shot: Number of samples for each class in the shot-shot classification task
            way: Number of classes in the shot-shot classification task
            query: Number of query samples for each class in the shot-shot classification task

        # Returns
            prepare_nshot_task_: A Callable that processes a few shot tasks with specified shot, way and query
        """
        def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            """Create 0-way label and move to GPU.

            TODO: Move to arbitrary device
            """
            x, y = batch
            x = x.double().cuda()
            # Create dummy 0-(num_classes - 1) label, 每个 query 个, 请看create_nshot_task_label函数.
            y = create_nshot_task_label(way, query).cuda()
            return x, y
        return prepare_nshot_task_
    
    def split_instances_FEAT(self, data):
        # NB: Return idx, not instance.
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))
    def split_instances(self, data):
        if self.training:
            return data[:self.args.shot*self.args.way], data[self.args.shot*self.args.way:]
        else:
            return data[:self.args.eval_shot*self.args.eval_way], data[self.args.eval_shot*self.args.eval_way:]

    def forward(self, x, get_feature=False):
        # 做好embedding, 然后传support set, query set.
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0) # 删除维度为1的维度.

            instance_embs = self.encoder(x) # [instance num x feature num]
            # num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            # support_idx: [1 x n-shot x k-way], query_idx: [1 x query x way]
            # 这里idx都是按顺序的, 从support_idx 0 ~ (n-shot x k-way) - 1, 再 query_idx 从 (n-shot x k-way) ~ 结束.
            # support_idx, query_idx = self.split_instances_FEAT(x)

            support, query = self.split_instances(instance_embs)
            logits, logits_reg = self._forward(support, query)
            return logits, logits_reg

    def _forward(self, support, query):
        raise NotImplementedError('Suppose to be implemented by subclass')