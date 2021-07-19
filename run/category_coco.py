import json
import os.path as osp
data_root_path = '/user/zhangyk/ML-GCN/data/coco/data'

from tqdm import tqdm
import numpy as np

stype = 'train'
# annotations_file = json.load(open(osp.join(data_root_path, f'annotations/instances_{stype}2014.json')))
img_list = json.load(open(osp.join(data_root_path, f'{stype}_anno.json'), 'r'))
cat2idx = json.load(open(osp.join(data_root_path, 'category.json'), 'r'))
idx2cat = {v: k for k, v in cat2idx.items()}
sample_num = 10000

import random
random.seed(929)

single_labels_dict = {}
for i in tqdm(img_list):
    for j in i['labels']:
        if j not in single_labels_dict.keys():
            single_labels_dict[j] = 0
        else:
            single_labels_dict[j] += 1

import matplotlib.pyplot as plt

Y = sorted(single_labels_dict.values())[::-1]
X = np.arange(len(Y))
print(len(X))
plt.bar(X, Y)
plt.savefig('single_labels_result.png')

assert 0

labels_dict = {}
for i in tqdm(img_list):
    cur_l = sorted(i['labels'])
    emd = 0
    for j in cur_l:
        emd = emd * 100 + int(j)
    if emd not in labels_dict:
        labels_dict[emd] = 0
    else:
        labels_dict[emd] += 1

import matplotlib.pyplot as plt

Y = sorted(labels_dict.values())[17500:]
X = np.arange(len(Y))
plt.bar(X, Y)
plt.savefig('result.png',dpi = 400)
assert 0

for way in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    involve_list = []
    for _ in tqdm(range(sample_num)):
        sampled_class = random.sample(range(len(cat2idx)), way)
        cat2instance = {k: [] for k in sampled_class}
        def is_labels_in_sampled_class(l):
            for i in l:
                if not i in sampled_class:
                    return False
            return True

        in_true = 0
        for i in img_list:
            if is_labels_in_sampled_class(i['labels']):
                for j in i['labels']:
                    cat2instance[j].append(i['labels'])
                in_true += 1
        
        print(in_true)
        assert 0

        involve_list.append(in_true)

    print(np.mean(in_true))

assert 0

ratio_list, base_ratio_list = [], []

for _ in tqdm(range(sample_num)):
    base_class = set(random.sample(range(80), 64))
    novel_class = set(range(80)) - base_class

    def is_separate_category(l):
        base_flag, novel_flag = 0b1, 0b1
        for i in l:
            if i in base_class:
                base_flag = 0b0
            elif i in novel_class:
                novel_flag = 0b0
        return base_flag, novel_flag
        # return base_flag | novel_flag

    sep_true, sep_false = 0, 0
    base_true, novel_true = 0, 0

    for i in img_list:
        bf, nf = is_separate_category(i['labels'])
        if bf | nf:
            sep_true += 1
            if bf == 0b0 and nf == 0b1:
                base_true += 1
            elif nf == 0b0 and bf == 0b1:
                novel_true += 1
            else:
                assert 0
        else:
            sep_false += 1

    base_ratio_list.append(base_true / sep_true)
    ratio_list.append(sep_true / len(img_list))

print(np.mean(ratio_list))
print(np.mean(base_ratio_list))