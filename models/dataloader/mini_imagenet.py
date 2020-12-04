import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from skimage import io
import pandas as pd
import os

sys.path.append("../../")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from config import DATA_PATH
from models.core import NShotTaskSampler
THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(DATA_PATH, 'data/mini_imagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/mini_imagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')

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
        # trick 1

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
        
        for l in tqdm(lines, ncols=64):
            img_name, class_name = l.split(',')
            path = osp.join(IMAGE_PATH, img_name)
            images.append({
                'stype': stype,
                'class_name': class_name,
                'filepath': osp.join(IMAGE_PATH, img_name)
            })
        return images

def dataloader_main(args):
    train = eval(args.dataset)(stype, args)
    train_taskloader = DataLoader(
        train,
        batch_sampler=NShotTaskSampler(train, episodes_per_epoch, args.shot, args.way, args.query),
        num_workers=4
    )

    return train_taskloader
