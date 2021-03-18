# import argparse

# import os
# import os.path as osp

# import shutil
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim import Optimizer

# from models.few_shot.helper import PrepareFunc
# from models.few_shot.protonet import PretrainClassifier

# from models.utils import pprint, set_gpu, set_logger, Timer, Averager, mkdir, get_command_line_parser, preprocess_args
# from models.dataloader.mini_imagenet import MiniImageNet
# from models.metrics import NAMED_METRICS
# from models.callbacks import EvaluateFewShot, CallbackList

# from typing import Callable

# if __name__ == '__main__':
#     parser = get_command_line_parser()
#     args = preprocess_args(parser.parse_args())
#     args.z_comment = 'Pretrain - ' + args.z_comment

#     pprint(vars(args))

#     prepare_handle = PrepareFunc(args)
#     (train_loader, train_num_classes), (val_loader, val_num_classes), (test_loader, _) \
#          = prepare_handle.get_dataloader(option=0b011)
#     loss_fn = prepare_handle.prepare_loss_fn()
    
#     model = PretrainClassifier(args, train_num_classes)
#     if 'Conv' in args.backbone_class:
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
#     elif 'Res' in args.backbone_class:
#         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
#     else:
#         raise ValueError('No Such Encoder')

#     if torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True
#         if ',' in self.args.gpu:
#             model.encoder = torch.nn.DataParallel(model.encoder)
#         model = model.to(torch.device('cuda'))
#         loss_fn = loss_fn.to(torch.device('cuda'))
    

#     def fit_handle(
#         model: nn,
#         optimizer: Optimizer,
#         loss_fn: Callable
#         ):
#         def core(
#             x: torch.Tensor,
#             y: torch.Tensor,
#             prefix: str = 'train_'
#             ):
#             if prefix == 'train_':
#                 # Zero gradients
#                 model.train()
#                 optimizer.zero_grad()
#             else:
#                 model.eval()

#             logits, reg_logits = model.forward_handle(x, prefix)

#             loss = loss_fn(logits, y)

#             if prefix == 'train_':
#                 # Take gradient step
#                 loss.backward()
#                 optimizer.step()
#             return logits, reg_logits, loss
#         return core
    
#     fit = fit_handle(model, optimizer, loss_fn)

#     evaluate_handle = EvaluateFewShot(
#         val_loader=val_loader,
#         test_loader=test_loader,
#         val_prepare_batch=model.prepare_kshot_task(args.val_way, args.val_query, args.meta_batch_size),
#         test_prepare_batch=model.prepare_kshot_task(args.test_way, args.test_query, args.meta_batch_size),
#         metric_name='ca_acc',
#         eval_fn=fit,
#         test_interval=args.test_interval,
#         metric_func=args.metric_func,
#         max_epoch=args.max_epoch,
#         model_filepath=args.model_filepath,
#         model_filepath_test_best=args.model_filepath_test_best,
#         monitor='ca_acc',
#         save_best_only=True,
#         mode='max',
#         simulation_test=False,
#         verbose=args.verbose,
#         epoch_verbose=args.epoch_verbose
#     )

#     logger_filename = osp.abspath(osp.dirname(__file__)) + f'{args.logger_filename}/process/{args.params_str}.log'
#     logger = set_logger(logger_filename, 'pretrain_logger')

#     callbacks = CallbackList([evaluate_handle])
#     callbacks.set_model_and_logger(model, logger)

#     def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
#         state = {'epoch': epoch + 1,
#                  'args': args,
#                  'state_dict': model.state_dict(),
#                  'trlog': trlog,
#                  'val_acc_dist': trlog['max_acc_dist'],
#                  'val_acc_sim': trlog['max_acc_sim'],
#                  'optimizer' : optimizer.state_dict(),
#                  'global_count': global_count}
    
#         torch.save(state, osp.join(args.pretrain_model_filepath, filename))
#         if is_best:
#             shutil.copyfile(osp.join(args.pretrain_model_filepath, filename), osp.join(args.pretrain_model_filepath, 'model_best.pth.tar'))

#     # if args.resume == True:
#     #     # load checkpoint
#     #     state = torch.load(osp.join(args.pretrain_model_filepath, 'model_best.pth.tar'))
#     #     init_epoch = state['epoch']
#     #     resumed_state = state['state_dict']
#     #     # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
#     #     model.load_state_dict(resumed_state)
#     #     trlog = state['trlog']
#     #     optimizer.load_state_dict(state['optimizer'])
#     #     initial_lr = optimizer.param_groups[0]['lr']
#     #     global_count = state['global_count']
#     # else:
#     #     init_epoch = 1     
#     #     initial_lr = args.lr
#     #     global_count = 0

#     timer = Timer()

#     if args.verbose:
#         logger.info('Begin training...')
#     if not args.epoch_verbose:
#         print('Begin training...')

#     callbacks.on_train_begin()
#     for epoch in range(1, args.max_epoch + 1):
#         epoch_logs = {}

#         # # refine the step-size
#         # if epoch in args.schedule:
#         #     initial_lr *= args.gamma
#         #     for param_group in optimizer.param_groups:
#         #         param_group['lr'] = initial_lr

#         tl, ta = Averager(), Averager()
#         for batch_index, batch in enumerate(train_loader):
#             if torch.cuda.is_available():
#                 x, y = [_.to(torch.device('cuda')) for _ in batch]
#                 y = y.type(torch.cuda.LongTensor)
#             else:
#                 x, y = batch
#                 y = y.type(torch.LongTensor)

#             logits, reg_logits, loss = fit(x=x, y=y, prefix='train_')
#             acc = NAMED_METRICS[args.metric_func](logits, y)                

#             tl.add(loss.item())
#             ta.add(acc)

#         tl = tl.item()
#         ta = ta.item()

#         if args.verbose:
#             print('Epoch {}: loss={:.4f} acc={:.4f}'.format(epoch, tl, ta))

#         callbacks.on_epoch_end(epoch, epoch_logs)