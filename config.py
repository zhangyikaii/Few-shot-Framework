import os.path as osp
DATA_PATH = osp.normpath('/mnt/data3/lus/zhangyk/data')
if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')