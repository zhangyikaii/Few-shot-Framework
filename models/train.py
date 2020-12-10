import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import Callable, List, Union
import os.path as osp

from models.callbacks import (
    DefaultCallback,
    ProgressBarLogger,
    CallbackList,
    Callback,
    EvaluateFewShot,
    ModelCheckpoint,
    CSVLogger
)

from models.dataloader.mini_imagenet import get_dataloader
from models.utils import PrepareFunc, set_logger
from models.sampler import prepare_nshot_task

from models.metrics import (
    NAMED_METRICS,
    categorical_accuracy
)

def gradient_step(model: Module, optimizer: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimizer: optimizer to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    return loss, y_pred

class Trainer(object):
    def __init__(self, args):
        self.logger = set_logger(args, 'train_logger')
        """
        准备 Dataloader
        """
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        """
        准备 Model, Optimizer, loss_fn, callbacks
        """
        prepare_handle = PrepareFunc(args)
        # 前向传播所需:
        # ( model 有且仅有这一个, callbacks 基类派生类都是共享这一个model. 下面其他东西也是这样存储 )
        self.model, self.para_model = prepare_handle.prepare_model()
        self.optimizer, self.lr_scheduler = prepare_handle.prepare_optimizer(self.model)
        self.loss_fn = prepare_handle.prepare_loss_func()
        self.total_loss = 0
        # 接下来要准备fit函数之前的所有东西, 包括callbacks.
        # 记录数据所需:
        # 这里的params统一传到基类成员, 所有派生类共享. 注意这里一定要精简.
        self.metrics = ['categorical_accuracy']
        self.prepare_batch = prepare_nshot_task(args.eval_shot, args.eval_way, args.eval_query)
        self.verbose = args.verbose
        self.max_epoch = args.max_epoch
        # self.params = {
        #     'max_epoch': args.max_epoch,
        #     'verbose': args.verbose,
        #     'metrics': (self.metrics or []),
        #     'prepare_batch': prepare_nshot_task(args.eval_shot, args.eval_way, args.eval_query),
        #     'loss_fn': self.loss_fn,
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': self.lr_scheduler
        # }

        # args 是一个参数集合, 期望在这里分模块, 对每个类 对应特定的功能, 类的参数也要**具体化**, 这样才可以一层层分解, 较好.

        self.metric_name = f'{args.eval_shot}-shot_{args.eval_way}-way_acc'
        self.model_filepath = f'/mnt/data3/lus/zhangyk/models/proto_nets/{args.params_str}.pth'
        self.evaluate_handle = EvaluateFewShot(
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            prepare_batch=self.prepare_batch,
            metric_name=self.metric_name,
            eval_fn=self.fit_handle,
            verbose=self.verbose,
            simulation_test=True
        )
        
        callbacks = [
            self.evaluate_handle,
            ModelCheckpoint(
                model_filepath=self.model_filepath,
                monitor=self.metric_name,
                save_best_only=True,
                mode='max',
                verbose=self.verbose
            ),
            CSVLogger(
                osp.abspath(osp.dirname(osp.dirname(__file__))) + f'/logs/{args.model_class}/{args.params_str}.csv',
                separator=',',
                append=False
            )
        ]

        # LearningRateScheduler 最好直接在fit函数里面传一个lr_scheduler, 直接step吧. 看FEAT.
        self.callbacks = CallbackList((callbacks or [])
                                     + [ProgressBarLogger(length=len(self.train_loader), verbose=self.verbose), ])

    def batch_metrics(self, logits, y, batch_logs):
        self.model.eval()
        for m in self.metrics:
            if isinstance(m, str):
                batch_logs[m] = NAMED_METRICS[m](y, logits)
            # else:
            #     # Assume metric is a callable function
            #     batch_logs = m(y, logits)

        return batch_logs


    def fit_handle(self,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   train: bool = True):
        if train:
            # Zero gradients
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        
        logits, reg_logits = self.model(x)
        loss = self.loss_fn(logits, y)

        if train:
            # Take gradient step
            loss.backward()
            self.optimizer.step()
        return logits, reg_logits, loss

    def lr_handle(self, epoch, epoch_logs):
        self.lr_scheduler.step()

    def test(self):
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.evaluate_handle.predict_log(self.max_epoch, self.test_loader, 'test_')

    def fit(self):
        # Determine number of samples:
        batch_size = self.train_loader.batch_size

        self.callbacks.set_model(self.model)

        if self.verbose:
            print('Begin training...')

        self.callbacks.on_train_begin()
        
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/model')

        for epoch in range(1, self.max_epoch+1):
            self.callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            for batch_index, batch in enumerate(self.train_loader):
                batch_logs = {}

                self.callbacks.on_batch_begin(batch_index, batch_logs)

                x, y = self.prepare_batch(batch)

                logits, reg_logits, loss = self.fit_handle(x, y)
                batch_logs['loss'] = loss.item()
                # Loops through all metrics
                batch_logs = self.batch_metrics(logits, y, batch_logs)

                self.callbacks.on_batch_end(batch_index, batch_logs)
            # Run on epoch end
            # 注意这个 epoch_logs 是共享变量, 在callbacks里面的类传递的!
            self.lr_handle(epoch, epoch_logs)
            self.callbacks.on_epoch_end(epoch, epoch_logs)

        # Run on train end
        if self.verbose:
            print('Finished.')

        self.callbacks.on_train_end()


