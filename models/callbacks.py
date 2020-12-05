from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict, Iterable
from typing import List, Iterable, Callable, Tuple
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


# 父类共享方法框架. 但注意这里只是方法, 不存模型.
class Callback(object):
    def __init__(self):
        self.params = None
    def set_params(self, params):
        self.params = params

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



class EvaluateFewShot(Callback):
    """Evaluate a network on an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 num_tasks: int,
                 k_way: int,
                 n_shot: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimizer = self.params['optimizer']

    # 在测试数据上val: 注意这里进来是evaluation文件夹下的数据, 前面训练的是background文件夹下面的数据.
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            # 这里eval_fn就是该类(在callbacks(list))初始化时传进来的函数, 在/few_shot/下. \
            #   比如 matching_net_episode.

            # on_epoch_end 这里的: \
            #   这里看 诸如matching_net_episode 的传参是什么, 请注意传的model就是 set_model 里面设置的model, \
            #   实际上就是! fit 函数时传进来的model. 这个model就是在models.py里面定义的.
            # 注意这里的传参逻辑一定要搞清楚.
            
            # 注意这里就是测试过程了呀, 完全在evaluation文件夹下数据上测, 注意train=False, 虽然还是用 proto_net_episode.
            loss, y_pred = self.eval_fn(
                self.model,
                self.optimizer,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen

        # 注意! 最后的测试准确率就在这里了, 这是在validation上的!
        logs[self.metric_name] = totals[self.metric_name] / seen



