import torch
import os.path as osp
from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision import transforms
import numpy as np
from skimage import io
import pandas as pd
import os, sys

sys.path.append("../../")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from config import DATA_PATH
from models.sampler import ShotTaskSampler
THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(DATA_PATH, 'mini_imagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/mini_imagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')

# for test
class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


class MiniImageNet(Dataset):
    def __init__(self, stype, args):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            stype: Whether the dataset represents the train, val, test set
        """
        if stype not in ('train', 'val', 'test'):
            raise(ValueError, 'stype must be one of (train, val, test)')
        self.stype = stype
        # index_stype 返回在该stype的字典: {stype名, 类名, 路径}
        self.df = pd.DataFrame(self.index_stype(self.stype))

        # 添加新的idx列:
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # 添加新的class_id列:
        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # 构建 dataID -> filepath/class_id 的字典:
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # 数据预处理:
        # Setup transforms
        # trick

        ### few-shot code:
        # self.transform = transforms.Compose([
        #     transforms.CenterCrop(224),
        #     transforms.Resize(84),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        ##################

        image_size = 84
        if args.augment and stype == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    # 通过下标访问时, 通过 dataID -> filepath/class_id 的字典, 返回图片, label.
    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    # 从这里开始index索引数据集, 为取index操作准备.
    @staticmethod
    def index_stype(stype):
        """Index a stype by looping through all of its files and recording relevant information.

        # Arguments
            stype: Name of the stype

        # Returns
            A list of dicts containing information about all the image files in a particular stype of the
            miniImageNet dataset
        """
        images = []
        csv_path = osp.join(SPLIT_PATH, stype + '.csv')

        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        
        for l in lines:
            img_name, class_name = l.split(',')
            path = osp.join(IMAGE_PATH, img_name)
            images.append({
                'stype': stype,
                'class_name': class_name,
                'filepath': osp.join(IMAGE_PATH, img_name)
            })
        return images

def get_dataloader(args):
    def taskloader(stype, args, shot, way, query, episodes_per_epoch):
        dataset = eval(args.dataset)(stype, args)
        taskloader = DataLoader(
            dataset,
            batch_sampler=ShotTaskSampler(
                dataset=dataset,
                episodes_per_epoch=episodes_per_epoch,
                shot=shot,
                way=way,
                query=query,
                num_tasks=args.meta_batch_size,
            ),
            num_workers=4,
            pin_memory=True
        )
        return taskloader

    train_taskloader, val_taskloader, test_taskloader = \
        taskloader('train', args, args.shot, args.way, args.query, args.episodes_per_epoch), \
        taskloader('val', args, args.shot, args.way, args.query, args.episodes_per_val_epoch), \
        taskloader('test', args, args.eval_shot, args.eval_way, args.eval_query, args.episodes_per_val_epoch)

    # return {'train': train_taskloader, 'val': val_taskloader, 'test': test_taskloader}
    return train_taskloader, val_taskloader, test_taskloader
