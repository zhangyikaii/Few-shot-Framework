from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch

# 因为Dataloader传入的数据集是 map-style datasets, (这也仅在这种下标数据集上使用.)
# 所以下面传入的NShotTaskSampler(batch_sampler参数) yields a list of batch indices.
class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        分 n*k 和 q*k, 并且是不相交的.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        请注意下面 num_tasks 和 episodes_per_epoch 的区别.
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        # print("episodes_per_epoch, num_tasks:", self.episodes_per_epoch, self.num_tasks)
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    # 随机sample k 个类:
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                    # print("fixed_tasks is None, episode_classes:", episode_classes)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                # support_k 字典, <key: k 个class, value: 每个class对应的n个数据.
                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    # 随机sample n 个样本, 在之前sample的k类的每类下.
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    # 随机sample q 个样本, 在之前sample的k类的每类下, 但是需要保证不和support有交集.
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(n=self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            # 请特别注意, batch里面的是按照顺序排的:
            #   n 个support 出现 k 次(因为k个不同类) + q 个query 出现 k 次.
            #   也就是 (n * k + q * k), 这样构成了一个task, 这可以有很多个task, 请注意这里需要设置几个task, ProtoNet这里是一个.
            #   也就是 (n) 个这样一组是 一类, 达到 (n * k) 之后 (q) 个这样一组是一类.
            # episodes_per_epoch 决定了有多少个这样的多task训练.
            yield np.stack(batch)
