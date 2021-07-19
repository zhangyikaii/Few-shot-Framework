import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image

resize = 92
image_size = 84

# transforms_list = [
#     transforms.Resize(resize),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor()
#     ]


def set_seeds(torch_seed, cuda_seed, np_seed, random_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(cuda_seed)
    np.random.seed(np_seed)
    random.seed(random_seed)

set_seeds(929, 929, 929, 929)


import os.path as osp

to_tensor_transform = transforms.Compose([
    transforms.ToTensor()
    ])
reverse_transform = transforms.Compose([
    transforms.ToPILImage(),
    np.array,
    ])


filepath = '/data/zhangyk/data/mini_imagenet/images/n0212916500000256.jpg'
instance = Image.open(filepath).convert('RGB')

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
pad_length = 512

transforms_list = [
        transforms.RandomResizedCrop(size=100),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.6),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ]
# transforms_list = [
#     transforms.RandomResizedCrop(size=512),
#     transforms.ToTensor(),
#     ]
transform = transforms.Compose(
    transforms_list
    )

for i in range(8):
    instance_t = transform(instance)
    plt.imsave(
        filepath[filepath.rfind('/') + 1 : filepath.rfind('.jpg')] + f'_t_{i}.jpg',
        reverse_transform(instance_t)
        )

# 多个旋转:
# import math
# degrees_list = [30, 60, 90, 120, 150, 180]
# degrees_list = [i + 180 for i in degrees_list]
# pad_length = int(512 * math.sqrt(2))

# for cur_degree in degrees_list:
#     cur_radian = cur_degree / 180 * math.pi
#     transforms_list = [
#         transforms.Pad(pad_length, padding_mode='reflect'),
#         transforms.RandomRotation((cur_degree, cur_degree + 1)),
#         transforms.CenterCrop(512),
#         transforms.ToTensor(),
#         ]
#     transform = transforms.Compose(
#         transforms_list
#         )

#     instance_t = transform(instance)
#     plt.imsave(
#         filepath[filepath.rfind('/') + 1 : filepath.rfind('.jpg')] + f'_t_{cur_degree}.jpg',
#         reverse_transform(instance_t)
#         )



"""
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000403.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0314621900000028.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000256.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0211673800001135.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0314621900000667.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000557.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001244.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0314621900000251.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0211673800001090.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500001121.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001096.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0314621900000952.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0211673800001194.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000922.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000073.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0314621900000402.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0211673800000446.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000400.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000134.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000817.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000666.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000001067.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000940.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000029.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000649.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000652.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000827.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000314.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000762.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000904.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000001292.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000267.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000328.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000700.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001294.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000838.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001057.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000282.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000450.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000856.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000001033.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001014.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000139.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000256.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000753.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000597.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000676.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000192.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000130.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000360.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500001051.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400001037.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000689.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500000101.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000208.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400000092.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500001087.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000775.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400000022.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000042.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000560.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000464.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400000303.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000876.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000001207.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400000345.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0212916500000393.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0287152500001161.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0761348000000143.jpg
/mnt/data3/lus/zhangyk/data/mini_imagenet/images/n0244348400001164.jpg
"""