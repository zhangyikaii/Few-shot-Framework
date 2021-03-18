import os, sys
import subprocess
from datetime import datetime
import time

sys.path.append("../")

from models.utils import mkdir

out_dir = './output'
mkdir(out_dir)

class MyIter():
    def __init__(self, init_range=[], begin_element=0):
        self.range = init_range
        self.p = init_range.index(begin_element) if begin_element in init_range else 0
    def next(self):
        ret = self.range[self.p]
        self.p = (self.p + 1) % len(self.range)
        return ret

##################
# 在这里调整参数: #
##################

gpu_loop = MyIter(
    init_range = [
        # 0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15
    ],
    begin_element=10
)
# params_loop = [0.1, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005]
# params_loop = [3, 6, 10, 30, 50, 70, 90]
params_loop = [0.1, 0.3, 0.5, 0.7, 0.9]
params = 'gamma'

##################


if __name__ == '__main__':
    # os.system('source activate zykycy')
    command = \
'nohup \
python ../main.py \
--meta_batch_size 1 \
--data_path /mnt/data3/lus/zhangyk/data/ye \
--max_epoch 200 \
--gpu {gpu} \
--model_class ProtoNet \
--distance l2 \
--backbone_class ConvNet \
--dataset MiniImageNet \
--train_way 5 --val_way 5 --test_way 5 \
--train_shot 5 --val_shot 5 --test_shot 5 \
--train_query 15 --val_query 15 --test_query 15 \
--logger_filename /logs \
--temperature 64 \
--lr_scheduler step \
--lr 0.001 --lr_mul 10 \
--step_size {params} \
--gamma 0.5 \
--val_interval 1 \
--test_interval 0 \
--loss_fn nn-cross_entropy \
--metrics categorical_accuracy \
--time_str \"{time}\" \
--verbose \
> \"{out}\" 2>&1 &'

    for params_name in params_loop:
        cur_gpu = gpu_loop.next()
        cur_time = datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
        out_name = '{time} {params}-{params_name} {gpu}.out'.format(
            time=cur_time,
            params=params,
            params_name=params_name,
            gpu=cur_gpu
        )
        os.system(command.format(
            params=params_name,
            gpu=cur_gpu,
            time=cur_time,
            out=os.path.join(out_dir, out_name)
        ))
        time.sleep(10)