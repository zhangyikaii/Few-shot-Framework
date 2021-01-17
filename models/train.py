import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from typing import Callable, List, Union
import os
import os.path as osp
import json
import tqdm

from models.callbacks import (
    ProgressBarLogger,
    CallbackList,
    Callback,
    EvaluateFewShot,
    CSVLogger
)
from models.few_shot.protonet import fit_handle as ProtoNet_fit_handle
from models.few_shot.maml import fit_handle as Maml_fit_handle

from models.dataloader.mini_imagenet import get_dataloader
from models.few_shot.helper import PrepareFunc
from models.utils import set_logger

from models.metrics import (
    NAMED_METRICS,
    categorical_accuracy
)

class Trainer(object):
    def __init__(self, args):
        torch.backends.cudnn.benchmark = True
        self.logger_filename = osp.abspath(osp.dirname(osp.dirname(__file__))) + f'{args.logger_filename}/process/{args.params_str}.log'
        self.result_filename = osp.abspath(osp.dirname(osp.dirname(__file__))) + f'{args.logger_filename}/result/{args.params_str}.csv'

        self.logger = set_logger(self.logger_filename, 'train_logger')
        for k in sorted(vars(args).keys()):
            self.logger.info(k + ': %s' % str(vars(args)[k]))

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
        self.model = prepare_handle.prepare_model()
        self.optimizer, self.lr_scheduler = prepare_handle.prepare_optimizer(self.model)
        self.loss_fn = prepare_handle.prepare_loss_fn()
        self.fit_handle = eval(args.model_class + '_fit_handle')(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn
        )

        self.total_loss = 0
        # 接下来要准备fit函数之前的所有东西, 包括callbacks.
        # 记录数据所需:
        # 这里的params统一传到基类成员, 所有派生类共享. 注意这里一定要精简.
        self.metrics = 'categorical_accuracy'
        self.train_prepare_batch = self.model.prepare_kshot_task(args.shot, args.way, 
            args.query, args.meta_batch_size)
        self.verbose, self.epoch_verbose = args.verbose, args.epoch_verbose
        self.max_epoch = args.max_epoch
        # self.params = {
        #     'max_epoch': args.max_epoch,
        #     'verbose': args.verbose,
        #     'metrics': (self.metrics or []),
        #     'prepare_batch': prepare_kshot_task(args.test_shot, args.test_way, args.test_query),
        #     'loss_fn': self.loss_fn,
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': self.lr_scheduler
        # }

        # args 是一个参数集合, 期望在这里分模块, 对每个类 对应特定的功能, 类的参数也要**具体化**, 这样才可以一层层分解, 较好.

        self.metric_name = 'ca_acc'
        self.model_filepath = args.model_filepath
        self.train_mode = args.train_mode
        
        self.evaluate_handle = EvaluateFewShot(
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            val_prepare_batch=self.model.prepare_kshot_task(args.shot, args.val_way, args.query, args.meta_batch_size),
            test_prepare_batch=self.model.prepare_kshot_task(args.test_shot, args.test_way, args.test_query, args.meta_batch_size),
            metric_name=self.metric_name,
            eval_fn=self.fit_handle,
            test_interval=args.test_interval,
            model_filepath=self.model_filepath,
            model_filepath_test_best=args.model_filepath_test_best,
            monitor=self.metric_name,
            save_best_only=True,
            mode='max',
            simulation_test=False,
            verbose=self.verbose
        )
        
        callbacks = [
            self.evaluate_handle,
            CSVLogger(
                self.result_filename,
                metric_name=self.metric_name,
                separator=',',
                append=False
            ),
            ProgressBarLogger(length=len(self.train_loader), verbose=self.epoch_verbose)
        ]

        # LearningRateScheduler 最好直接在fit函数里面传一个lr_scheduler, 直接step吧. 看FEAT.
        self.callbacks = CallbackList((callbacks or []))
        self.callbacks.set_model_and_logger(self.model, self.logger)

        """
        meta
        """
        self.meta = args.meta

    def delete_logs(self):
        os.remove(self.logger_filename)
        os.remove(self.result_filename)

    def batch_metrics(self, logits, y, batch_logs):
        self.model.eval()
        with torch.no_grad():
            batch_logs[self.metrics] = NAMED_METRICS[self.metrics](y, logits)
            # # 迭代更新每一个度量.
            # for m in self.metrics:
            #     if self.meta:
            #         # if meta, logits is acc.
            #         batch_logs[m] = logits
            #         return batch_logs
            #     batch_logs[m] = NAMED_METRICS[m](y, logits)
            #     # else:
            #     #     # Assume metric is a callable function
            #     #     batch_logs = m(y, logits)

        return batch_logs

    def lr_handle(self, epoch, epoch_logs):
        self.lr_scheduler.step()

    def test(self):
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.logger.info(f'Testing model: {self.model_filepath}')
        return self.evaluate_handle.predict_log(self.max_epoch, self.test_loader, 'test_')

    def fit(self):
        if not self.train_mode:
            return
        # Determine number of samples:
        batch_size = self.train_loader.batch_size

        if self.verbose:
            # print('Begin training...')
            self.logger.info('Begin training...')

        self.callbacks.on_train_begin()

        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/model')

        if not self.epoch_verbose:
            pbar = tqdm.trange(1, self.max_epoch+1)
        else:
            pbar = range(1, self.max_epoch+1)
        for epoch in pbar:
            self.callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            for batch_index, batch in enumerate(self.train_loader):
                batch_logs = {}

                self.callbacks.on_batch_begin(batch_index, batch_logs)

                x, y = self.train_prepare_batch(batch)

                logits, reg_logits, loss = self.fit_handle(x=x, y=y, prefix='train_')
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
            # print('Finished.')
            self.logger.info('Finished')

        self.callbacks.on_train_end()


