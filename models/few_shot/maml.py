import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F

import numpy as np

from models.metrics import pairwise_distances
from models.utils import create_nshot_task_label
from models.few_shot.base import FewShotModel
from models.backbone.blocks import functional_conv_block
from models.backbone.convnet import conv_block

from typing import Callable, Tuple
from collections import OrderedDict

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_



def fit_handle_another(
    model: nn,
    optimizer: Optimizer,
    loss_fn: Callable
    ):

    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        train: bool = True
        ):
        """
        create_graph: 构造导数图, 允许计算高阶导数的乘积.
        If you have to use this function, make sure to reset the .grad fields of your 
        parameters to None after use to break the cycle and avoid the leak.
        """
        args = model.args
        data_shape = x.shape[2:]
        create_graph = (True if args.order == 2 else False) and train

        task_gradients = []
        task_losses = []
        task_predictions = []
        for meta_batch in x:
            # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
            # Hence when we iterate over the first  dimension we are iterating through the meta batches
            x_task_train = meta_batch[:args.shot * args.way]
            x_task_val = meta_batch[args.shot * args.way:]

            # Create a fast model using the current meta model weights
            fast_weights = OrderedDict(model.named_parameters())

            # Train the model for `inner_train_steps` iterations
            for inner_batch in range(args.inner_train_steps):
                # Perform update of model weights
                y = create_nshot_task_label(args.way, args.shot).cuda()
                logits = model.functional_forward(x_task_train, fast_weights)
                loss = loss_fn(logits, y)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

                # Update weights manually
                fast_weights = OrderedDict(
                    (name, param - args.inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )


            # Do a pass of the model on the validation data from the current task
            y = create_nshot_task_label(args.way, args.query).to(torch.device("cuda"))
            logits = model.functional_forward(x_task_val, fast_weights)
            loss = loss_fn(logits, y)
            loss.backward(retain_graph=True)

            # Get post-update accuracies
            y_pred = logits.softmax(dim=1)
            task_predictions.append(y_pred)

            # Accumulate losses and gradients
            task_losses.append(loss)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
            task_gradients.append(named_grads)

        if args.order == 1:
            if train:
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                    for k in task_gradients[0].keys()}
                hooks = []
                for name, param in model.named_parameters():
                    hooks.append(
                        param.register_hook(replace_grad(sum_task_gradients, name))
                    )

                model.train()
                optimizer.zero_grad()
                # Dummy pass in order to create `loss` variable
                # Replace dummy gradients with mean task gradients using hooks
                logits = model(torch.zeros((args.way, ) + data_shape).to(torch.device("cuda"), dtype=torch.double))
                loss = loss_fn(logits, create_nshot_task_label(args.way, 1).to(torch.device("cuda")))
                loss.backward()
                optimizer.step()

                for h in hooks:
                    h.remove()

            return torch.stack(task_losses).mean(), torch.cat(task_predictions)

        elif args.order == 2:
            model.train()
            optimizer.zero_grad()
            meta_batch_loss = torch.stack(task_losses).mean()

            if train:
                meta_batch_loss.backward()
                optimizer.step()

            return meta_batch_loss, torch.cat(task_predictions)
        else:
            raise ValueError('Order must be either 1 or 2.')

    return core

def fit_handle(
    model: nn,
    optimizer: Optimizer,
    loss_fn: Callable
    ):

    def core(
        x: torch.Tensor,
        y: torch.Tensor,
        train: bool = True
        ):
        """
        create_graph: 构造导数图, 允许计算高阶导数的乘积.
        If you have to use this function, make sure to reset the .grad fields of your 
        parameters to None after use to break the cycle and avoid the leak.
        """

        args = model.args
        
        # sptsz = args.shot * args.way = 5, qrysz = args.query * args.way = 25.
        y_spt = create_nshot_task_label(args.way, args.shot).to(torch.device("cuda"))
        y_qry = create_nshot_task_label(args.way, args.query).to(torch.device("cuda"))

        losses_q = [0 for _ in range(args.inner_train_steps + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(args.inner_train_steps + 1)]

        for meta_batch in x:
            support = meta_batch[:args.shot * args.way]
            query = meta_batch[args.shot * args.way:]

            # support: (sptsz x C x H x W) = (5 x 3 x H x W)
            logits = model(support) # logits: (sptsz x way) = (5 x 5)
            loss = loss_fn(logits, y_spt)
            grad = torch.autograd.grad(loss, model.parameters())
            fast_weights = OrderedDict(
                (name, param - args.inner_lr * grad)
                for ((name, param), grad) in zip(model.named_parameters(), grad)
            )
            with torch.no_grad():
                logits_q = model(query) # 注意这里是直接用网络的参数.
                # logits_q: (qrysz x way) 像上面一样.
                loss_q = loss_fn(logits_q, y_qry)
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct
            with torch.no_grad():
                logits_q = model.functional_forward(query, fast_weights)
                # logits_q: (qrysz x way) 像上面一样.
                loss_q = loss_fn(logits_q, y_qry)
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct

            for inner_batch in range(1, args.inner_train_steps):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = model.functional_forward(support, fast_weights)
                loss = loss_fn(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights.values())
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = OrderedDict(
                    (name, param - args.inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), grad)
                )

                logits_q = model.functional_forward(query, fast_weights)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = loss_fn(logits_q, y_qry)
                losses_q[inner_batch + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                    corrects[inner_batch + 1] = corrects[inner_batch + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        task_num = x.shape[0]
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        if train:
            model.train()
            optimizer.zero_grad()
            loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())

        accs = np.array(corrects) / (args.way * args.query * task_num)

        return accs, None, loss_q

        # task_gradients, task_losses, task_predictions = [], [], []

        # for meta_batch in x:
        #     # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        #     # Hence when we iterate over the first  dimension we are iterating through the meta batches
            

        #     # Create a fast model using the current meta model weights
        #     fast_weights = OrderedDict(model.named_parameters())

        #     # Train the model for `inner_train_steps` iterations
        #     for inner_batch in range(args.inner_train_steps):
        #         # Perform update of model weights
        #         y = create_nshot_task_label(args.way, args.shot).cuda()
        #         logits = model.functional_forward(x_task_train, fast_weights)
        #         loss = loss_fn(logits, y)
        #         gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

        #         # Update weights manually
        #         fast_weights = OrderedDict(
        #             (name, param - args.inner_lr * grad)
        #             for ((name, param), grad) in zip(fast_weights.items(), gradients)
        #         )

        #     # Do a pass of the model on the validation data from the current task
        #     y = create_nshot_task_label(args.way, args.query).cuda()
        #     logits = model.functional_forward(x_task_val, fast_weights)
        #     loss = loss_fn(logits, y)
        #     loss.backward(retain_graph=True)

        #     # Get post-update accuracies
        #     y_pred = logits.softmax(dim=1)
        #     task_predictions.append(y_pred)

        #     # Accumulate losses and gradients
        #     task_losses.append(loss)
        #     gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
        #     named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        #     task_gradients.append(named_grads)

        # reg_logits = None
        # if args.order == 1:
        #     if train:
        #         sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
        #                             for k in task_gradients[0].keys()}
        #         hooks = []
        #         for name, param in model.named_parameters():
        #             hooks.append(
        #                 param.register_hook(replace_grad(sum_task_gradients, name))
        #             )

        #         model.train()
        #         optimizer.zero_grad()
        #         # Dummy pass in order to create `loss` variable
        #         # Replace dummy gradients with mean task gradients using hooks
        #         logits = model(torch.zeros((args.way, ) + x.shape[2:]).to(torch.device("cuda"), dtype=torch.double))
        #         loss = loss_fn(logits, create_nshot_task_label(args.way, 1).to(torch.device("cuda")))
        #         loss.backward()
        #         optimizer.step()

        #         for h in hooks:
        #             h.remove()

        #     return torch.cat(task_predictions), reg_logits, torch.stack(task_losses).mean(), 

        # elif args.order == 2:
        #     model.train()
        #     optimizer.zero_grad()
        #     meta_batch_loss = torch.stack(task_losses).mean()

        #     if train:
        #         meta_batch_loss.backward()
        #         optimizer.step()

        #     return torch.cat(task_predictions), reg_logits, meta_batch_loss
        # else:
        #     raise ValueError('Order must be either 1 or 2.')

    return core


# 可能不能继承 FewShotModel.
# 没错是的.
class MAML(nn.Module):
    def __init__(self, args):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `backbone` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(MAML, self).__init__()
        self.args = args # 等等优化器算参数记得拿掉.
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(1600, args.way)

    def prepare_nshot_task(self, shot: int, way: int, query: int, meta_batch_size: int) -> Callable:
        def prepare_meta_batch_(batch):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients
            num_input_channels = 3 # MiniImageNet: 三通道图片.
            x = x.reshape(meta_batch_size, shot*way + query*way, num_input_channels, x.shape[-2], x.shape[-1])
            # 请注意在这里reshap之后的数据顺序是怎么样的.

            # x shape: [meta_batch_size x (shot*way + query*way) x 3 x 图片长 x 图片宽]
            # Move to device
            x = x.double().cuda()
            # Create label
            y = create_nshot_task_label(way, query).cuda().repeat(meta_batch_size)
            # y shape: 一列, way个query, 重复meta_batch_size次, 一共 query*way*meta_batch_size个.
            return x, y

        return prepare_meta_batch_

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # (N x ...)

        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        # .0.weight or .1.weight 0/1表示网络的第几层.
        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x