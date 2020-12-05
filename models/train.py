import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

from models.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback

from models.dataloader.mini_imagenet import get_dataloader
from models.utils import PrepareFunc

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
        """
        准备 Dataloader
        """
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        """
        准备 Model, Optimizer, loss_func, callbacks
        """
        prepare_handle = PrepareFunc(args)
        self.model, self.para_model = prepare_handle.prepare_model()
        self.optimizer, self.lr_scheduler = prepare_handle.prepare_optimizer(self.model)
        self.loss_fn = prepare_handle.prepare_loss_func()
        # 接下来要准备fit函数之前的所有东西, 包括callbacks.

        self.callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])

    def fit(self):
        """Function to abstract away training loop.

        The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
        common training functionality provided they are written as a subclass of voicemap.Callback (following the
        Keras API).

        # Arguments
            model: Model to be fitted.
            optimizer: optimizer to calculate gradient step from loss
            loss_fn: Loss function to calculate between predictions and outputs
            epochs: Number of epochs of fitting to be performed
            dataloader: `torch.DataLoader` instance to fit the model to
            prepare_batch: Callable to perform any desired preprocessing
            metrics: Optional list of metrics to evaluate the model with
            callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
                checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
            verbose: All print output is muted if this argument is `False`
            fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
                batches. For more complex training procedures (meta-learning etc...) you will need to write your own
                fit_function
            fit_function_kwargs: Keyword arguments to pass to `fit_function`
        """
        # Determine number of samples:
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size

        callbacks.set_model(model)
        callbacks.set_params({
            'num_batches': num_batches,
            'batch_size': batch_size,
            'verbose': verbose,
            'metrics': (metrics or []),
            'prepare_batch': prepare_batch,
            'loss_fn': loss_fn,
            'optimizer': optimizer
        })

        if verbose:
            print('Begin training...')

        callbacks.on_train_begin()
        
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter('runs/model')

        for epoch in range(1, epochs+1):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            for batch_index, batch in enumerate(dataloader):
                batch_logs = dict(batch=batch_index, size=(batch_size or 1))

                callbacks.on_batch_begin(batch_index, batch_logs)

                x, y = prepare_batch(batch)

                # writer.add_graph(model, (x,))

                # 注意这里的fit_function是根据用的模型来定的.
                loss, y_pred = fit_function(model, optimizer, loss_fn, x, y, **fit_function_kwargs)
                batch_logs['loss'] = loss.item()

                # Loops through all metrics
                batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

                callbacks.on_batch_end(batch_index, batch_logs)

            # Run on epoch end
            # 注意这个 epoch_logs 是共享变量, 在callbacks里面的类传递的!
            callbacks.on_epoch_end(epoch, epoch_logs)

        # Run on train end
        if verbose:
            print('Finished.')

        callbacks.on_train_end()
