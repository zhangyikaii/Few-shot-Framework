# test proto net ( for 2020-12-7 seminar )
import unittest

import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from torch.utils.data import DataLoader

from models.sampler import ShotTaskSampler
from models.dataloader.mini_imagenet import DummyDataset
from models.backbone.convnet import ConvNet
from models.few_shot.protonet import ProtoNet

class TestProtoNets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset(samples_per_class=1000, n_classes=20)

    def _test_n_k_q_combination(self, n, k, q):
        k_shot_taskloader = \
            DataLoader(self.dataset,
                       batch_sampler=ShotTaskSampler(self.dataset, 100, n, k, q))

        for batch in k_shot_taskloader:
            x, y = batch
            break

        support = x[:n * k]
        support_labels = y[:n * k]
        
        prototypes = ProtoNet.compute_prototypes(None, support, k, n)
        for i in range(k):
            self.assertEqual(
                support_labels[i * n],
                prototypes[i, 1],
                'Prototypes computed incorrectly!'
            )
    def test_compute_prototypes(self):
        test_combinations = [
            (1, 5, 5),
            (5, 5, 5),
            (1, 20, 5),
            (5, 20, 5)
        ]
        for n, k, q in test_combinations:
            self._test_n_k_q_combination(n, k, q)