from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict, Iterable
from models.metrics import categorical_accuracy
from typing import List, Iterable, Callable, Tuple
import warnings
import os
import csv
import io

from models.utils import mkdir

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

    def set_model(self, model):
        for callback in self.callbacks:
            # 每个对象设置到当前model.
            # 真正要看是怎么set_model的, 包括下面的 callback.操作函数 , 都要进到具体的对象里面看.

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


# 父类共享方法框架. 但注意这里只是方法, 不存模型.
class Callback(object):
    def __init__(self):
        self.model = None
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
    def __init__(self, metrics):
        super(DefaultCallback, self).__init__()
        self.metrics = metrics
        self.seen = 0
        self.totals = {}        

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
    def __init__(self, length, metrics, verbose=True):
        super(ProgressBarLogger, self).__init__()
        import torchvision
        self.length = length
        self.verbose = verbose
        self.metrics = metrics
    
    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = tqdm(total=self.length, desc='Epoch {}'.format(epoch))
        
        # TensorBoard Test:
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

        if self.verbose and self.seen < self.length:
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


# 未改, 最好在 eval_fn 调用处要改一下.
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
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 loss_fn: Callable,
                 optimizer: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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
            # 就相当于forward.
            loss, y_pred = self.eval_fn(
                x,
                y,
                train=False
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen

        # 注意! 最后的测试准确率就在这里了, 这是在validation上的!
        logs[self.metric_name] = totals[self.metric_name] / seen


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
    with the epoch number and the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_acc', verbose=True, save_best_only=True, mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less

        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # TODO: 这里的filepath没有嵌入epoch.
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)


class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        mkdir(filename[:filename.rfind('/')])
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'

        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    # 在这里将各种参数写入文件.
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        # row_dict 就是 csv 文件里记录的信息.
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


# class LearningRateScheduler(Callback):
#     """Learning rate scheduler.
#     # Arguments
#         schedule: a function that takes an epoch index as input
#             (integer, indexed from 0) and current learning rate
#             and returns a new learning rate as output (float).
#         verbose: int. 0: quiet, 1: update messages.
#     """

#     def __init__(self, schedule, verbose=True):
#         super(LearningRateScheduler, self).__init__()
#         self.schedule = schedule
#         self.verbose = verbose

#     def on_train_begin(self, logs=None):
#         self.optimizer = self.params['optimizer']

#     def on_epoch_begin(self, epoch, logs=None):
#         lrs = [self.schedule(epoch, param_group['lr']) for param_group in self.optimizer.param_groups]

#         if not all(isinstance(lr, (float, np.float32, np.float64)) for lr in lrs):
#             raise ValueError('The output of the "schedule" function '
#                              'should be float.')
#         self.set_lr(epoch, lrs)

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         if len(self.optimizer.param_groups) == 1:
#             logs['lr'] = self.optimizer.param_groups[0]['lr']
#         else:
#             for i, param_group in enumerate(self.optimizer.param_groups):
#                 logs['lr_{}'.format(i)] = param_group['lr']

#     def set_lr(self, epoch, lrs):
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             new_lr = lrs[i]
#             param_group['lr'] = new_lr
#             if self.verbose:
#                 print('Epoch {:5d}: setting learning rate'
#                       ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
