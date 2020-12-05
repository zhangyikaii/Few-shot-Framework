from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict, Iterable
import warnings
import os
import csv
import io

# init 函数传入的 callbacks 是一些对象(操纵类)集合.
# CallbackList 控制 callbacks 之前加入的所有对象(以list形式).
# 在 fit 函数里, 调用 on_epoch_begin 一次 => \
#   调用callbacks里面所有类的on_epoch_begin函数. (callbacks里面类都是Callback的子类, on_epoch_begin是虚函数, 这样继承的)

# 请看每个 /experiments/ 下的(特定Approach)文件, \
#   里面callbacks(list)所包含的类是大家公有的, 但是每个类初始化的函数(类内所使用的函数)是不同的(因方法而变的), \
#   这就在few_shot文件夹下了.

class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    """
    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            # 每个对象设置到当前model.
            # 真正要看是怎么set_model的, 包括下面的 callback.操作函数 , 都要进到具体的对象里面看.
            # [<few_shot.callbacks.DefaultCallback object at 0x7fdc196c1fd0>, 
            # <few_shot.core.EvaluateFewShot object at 0x7fdc196c1eb0>, 
            # <few_shot.callbacks.ModelCheckpoint object at 0x7fdc196c1ee0>, 
            # <few_shot.callbacks.LearningRateScheduler object at 0x7fdc196c1e80>, 
            # <few_shot.callbacks.CSVLogger object at 0x7fdc196c1f40>, 
            # <few_shot.callbacks.ProgressBarLogger object at 0x7fdc1baa50a0>]

            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


# 父类共享模型, 模型参数.
class Callback(object):
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

class DefaultCallback(Callback):
    """Records metrics over epochs by averaging over each batch.

    NB The metrics are calculated with a moving model
    """
    def on_epoch_begin(self, batch, logs=None):
        self.seen = 0
        self.totals = {}
        self.metrics = ['loss'] + self.params['metrics']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 1) or 1
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.metrics:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen


class ProgressBarLogger(Callback):
    """TQDM progress bar that displays the running average of loss and other metrics."""
    def __init__(self):
        super(ProgressBarLogger, self).__init__()
        import torchvision

    def on_train_begin(self, logs=None):
        self.num_batches = self.params['num_batches']
        self.verbose = self.params['verbose']
        self.metrics = ['loss'] + self.params['metrics']

    def on_epoch_begin(self, epoch, logs=None):
        self.target = self.num_batches
        self.pbar = tqdm(total=self.target, desc='Epoch {}'.format(epoch))
        
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('runs/epoch-' + str(epoch))
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        self.log_values = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1

        for k in self.metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        
        self.writer.add_scalar('Loss/train', self.log_values['loss'], self.seen)
        self.writer.add_scalar('Categorical Accuracy/train', self.log_values['categorical_accuracy'], self.seen)

        if self.verbose and self.seen < self.target:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        # Update log values
        self.log_values = {}
        for k in self.metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        if self.verbose:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

        self.pbar.close()

