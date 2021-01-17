from models.few_shot.base import FewShotModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.metrics import pairwise_distances
from torch.optim import Optimizer
from typing import Callable


# train/val的逻辑, 与模型方法相关.
# 写一个两层的函数, model之类先传
def fit_handle(
    model: nn,
    optimizer: Optimizer,
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
            optimizer.zero_grad()
        else:
            model.eval()

        logits, reg_logits = model(x, prefix)
        loss = loss_fn(logits, y)

        if prefix == 'train_':
            # Take gradient step
            loss.backward()
            optimizer.step()
        return logits, reg_logits, loss
    return core
    
class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def compute_prototypes(self, support: torch.Tensor, k: int, n: int) -> torch.Tensor:
        """Compute class prototypes from support samples.

        # Arguments
            support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
                dimension.
            k: int. "k-way" i.e. number of classes in the classification task
            n: int. "k-shot" of the classification task

        # Returns
            class_prototypes: Prototypes aka mean embeddings for each class
        """
        # Reshape so the first dimension indexes by class then take the mean
        # along that dimension to generate the "prototypes" for each class
        class_prototypes = support.reshape(k, n, -1).mean(dim=1)
        return class_prototypes

    def _forward(self, support, query, prefix):
        if prefix == 'train_':
            prototypes = self.compute_prototypes(self, support, self.args.way, self.args.shot)
        elif prefix == 'val_':
            prototypes = self.compute_prototypes(self, support, self.args.val_way, self.args.shot)
        elif prefix == 'test_':
            prototypes = self.compute_prototypes(self, support, self.args.test_way, self.args.test_shot)

        # Calculate squared distances between all queries and all prototypes
        # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
        distances = pairwise_distances(query, prototypes, self.args.distance, self.args.temperature)

        # Prediction probabilities are softmax over distances
        logits = -distances

        return logits, None

    def _forward_FEAT(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        # 根据下标把support取出来.
        # view后面是 [1 x k-shot x k-way], 就按照这样的 k-shot, k-way 取出了, 因为前面flatten之后就是没结构的规整的形式了.
        # 很精妙! 因为这样可以保留feature结构.
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        # support: [1 x k-shot x k-way x feature num]

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if True: # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch, Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits, None
