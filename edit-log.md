## TODO list:

---

+ 添加命令行参数: epoch num.

+ 添加命令行参数: batch_size, 似乎在传进fit函数的dataloader里面设置.

### FEAT 部分:

+ FEAT有添加args.balance对loss进行控制.
  
  few-shot比FEAT的优点就是似乎不只有 lr_scheduler, 但是这里最好改成像 optim.lr_scheduler.StepLR 一样的.  
  
  加入 load pre-trained model.
  
  一句 `model = eval(args.model_class)(args)` 决定什么模型, 可以借鉴.

该repo作者原话:
> In both cases the classes in the training and validation sets are disjoint. 
  I did not use the same training and validation splits as the original papers 
  as my goal is not to reproduce them down to their last minute detail.
> within a few % on the miniImageNet benchmark without having to perform any tuning of my own.
  但是实际上和原文差了两个点.

+ 关注一下miniImageNet图片大小:

  原本256x256, 使用RandomSizeCrop裁剪, 84x84: Conv4/6 backbone, 224x224: ResNet backbone.

  这意味着改成resnet的backbone的话数据预处理也要改.

+ datasets.py 里 MiniImageNet 类的数据预处理时方差均值等是硬编码的.

+ 设置随机数种子是不是一个trick? `query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(n=self.q)` `core.py` 这里是否需要设置随机数种子. 前面的support, 它的类也是.

+ `ShotTaskSampler` 类太关键了, 所以之前没看导致relation net那边可能support/query写反了.

+ 2020-12-1: 下午看test部分.



## 备注:

---

models.py 里面的encoder()就是backbone.

+ experiments/ 下的每个方法文件, 相当于控制epoch的那层循环, 只是它用了fit函数(train.py) 封装实现.
    注意这里fit函数的顺序: 
      (epoch循环) on_epoch_begin -> (batch循环) on_batch_begin
      -> 参数 prepare_batch (函数) -> 参数 fit_function (函数, 就是例如matching_net_episode, few_shot/ 下面的)
      -> on_batch_end -> on_epoch_end

+ prepare_nshot_task函数(作为prepare_batch参数传入fit)运行结束后, 返回参数 y 并不是one-hot的.
    请注意, 该函数返回的:
    x: 是support和query的, 在接下来的 xxx_net_episode 函数中会被切分.
    y: 是仅有support的.

+ CrossEntropyLoss <=> Softmax-Log-NLLLoss

+ 封装 => 减少代码, 共享同样功能的代码 => 但是也减少了某些部分的灵活性. 从这个角度上看本项目的一个方法一个py是有益的.

+ 在main文件里, 一个技巧, 期望 dataset_class 根据参数变化(指向不同的类), 通过赋值即可.

+ `ShotTaskSampler` 决定了如何 generates batches of n-shot, k-way, q-query tasks.

  `--k-train 20 --n-train 1 --q-train 15` 请注意在训练时候的 `k, n, q` 是这样的.

+ `proto_net_episode` 的参数 `y` 是query set的label. 详见 `prepare_nshot_task`.

+ 纠正过来, 事实是: 训练时候 support 1-shot, query 20-shot.


## Trick

1. 野生数据预处理, FEAT里面似乎训练和测试还不一样, `RandomResizedCrop`, `ColorJitter` 之类的. 因为测试就只能按要求来?

2. `num_tasks` 之类的参数?

3. model & data: float32 or float64.

4. `fix_BN` ?

5. `reg_logits` ?


## Few-shot-Framework

+ 2020-12-4 21:35:51, 写完了Dataloader, 没测试. `MiniImageNet` 类还是要测一下, `augment` 参数没懂, 整个类和FEAT的还是不一样的.

+ `we use ADAM for ConvNet and SGD for other backbones` 为啥其他backbone就是sgd?

+ 2020-12-5 09:03:12 接下来构建`ProtoNet`

+ 注意, `prepare_nshot_task` 还要再写. 其实就是准备 `y` Label而已.

+ 期望的结果是增加模型后不用改动太多的文件, 即不同该`utils.py`文件

+ TODO: 在迭代内, progress bar 没有被更新, 是self.length那里出错吗?

+ 在forward函数内出错, 报错只会报: `= self.model(x)` 出错, 不会具体某一行.

+ `ValueError: too many values to unpack (expected 2)` 可能是返回了两个参数, 实际上接收那边只有一个.

+ 在 `nn` 模型内, 随着`model.eval()`, `self.traing` 在变化.

+ TODO: 注意在 `Dataloader` 那里不要分割support和query. 确认一下FEAT是怎么做的.

+ TODO: 添加终止程序后删除log file等.

+ 复制到 `E:\Few-shot-Framework` 再用git desktop更新.

+ 模型之间不同的有 (在各自模型文件里实现的): 1. `prepare_nshot_task` 函数, 2. `forward` 函数, 3. `fit_handle` 函数, 其中前两个在 `FewShotModel` 类内有一定的继承实现.

+ x = x.reshape(meta_batch_size, shot*way + query*way, num_input_channels, x.shape[-2], x.shape[-1]) 请注意在这里reshap之后的数据顺序是怎么样的.

+ TODO: MAML现在有个问题, 就是Dataloader那么episode数量train和val不一样.
  
  TODO: num_workers之类的参数搞一下, 试试多卡train.
  
  TODO: 解耦 ProtoNet 的每一步.

  TODO: few-shot 代码差不多了, 要专注于看FEAT代码.

  TODO: MAML episodes_per_val_epoch, inner_val_steps 等具体参数, 在测试时是多少, 还要再看看.

+ `lr_mul` 是做什么的?

+ TODO: 设置好随机种子, 这样结果不会太飘: 

  ```python
  torch.manual_seed(929)
  torch.cuda.manual_seed_all(929)
  np.random.seed(929)
  ```

+ `model.parameters()` 自然是不包括 model.args 的(自定义参数).

+ TODO:
  MAML 就是要搞清楚验证/测试的时候是怎么样的, 报错在:
  ```bash
  Traceback (most recent call last)| 0/40 [00:00<?, ?it/s]
    File "../main.py", line 24, in <module>
      trainer.fit()
    File "/home/lus/zhangyk/Few-shot-Framework/models/train.py", line 176, in fit
      self.callbacks.on_epoch_end(epoch, epoch_logs)
    File "/home/lus/zhangyk/Few-shot-Framework/models/callbacks.py", line 55, in on_epoch_end
      callback.on_epoch_end(epoch, logs)
    File "/home/lus/zhangyk/Few-shot-Framework/models/callbacks.py", line 236, in on_epoch_end
      self.predict_log(epoch, self.val_loader, 'val_', logs)
    File "/home/lus/zhangyk/Few-shot-Framework/models/callbacks.py", line 204, in predict_log
      logits, reg_logits, loss = self.eval_fn(
    File "/home/lus/zhangyk/Few-shot-Framework/models/few_shot/maml.py", line 56, in core
      # Train the model for `inner_train_steps` iterations
    File "/home/lus/anaconda3/envs/zykycy/lib/python3.8/site-packages/torch/autograd/__init__.py", line 190, in grad
      return Variable._execution_engine.run_backward(
  RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
  ```

## Summary

在这里理清整个过程:

```python
ProtoNet:
for epoch:
  for batch:
    过一遍episode里sample出来的所有. 代码在`yield np.stack(batch)`
    一个episode里有num_task轮.
    具体来说, 就是:
    (shot * way + query * way)
    (SupportSet + QuerySet)
    (1 * 5      + 15 * 5) = `N`
      不妨以上述数据(`len`)为例, 进到forward里的support+query:
        (80, 3, 84, 84) => (`N` x 通道数 `C` x 图片高 `H` x 图片宽 `W`)
        接下来encoder(backbone, 以ConvNet为例):
          结合下面`+ conv_block`看..
          四个 conv_block 后经过 AvgPool2d: (80 x 64 x 1 x 1), 这里 1 x 1 是经过很多层之后 H_{out} x W_{out} 的结果.
          最后 x.view(x.size(0), -1) 整理成两个维度 (N x feature_num(C_{out}))
        (80, 64) => (N x feature_num(C_{out})), 前1 * 5是support, 后15 * 5是query.
        support: (5 x 64), query: (75 x 64)
        接下来ProtoNet核心: `_forward`
          计算原型:
            --`compute_prototypes`-> (way x shot x feature): (5 x 1 x 64), 先分way.
              --`mean`-> (way x feature): (5 x 64), 计算way个原型, 在shot的维度上取mean.
          计算每个query 到 每个原型的距离:
            (75 x 64) 和 (5 x 64) 算距离, 每个都是 64长度 的vector被拿出来算.
            --`pairwise_distances`-> (75 x 5), 每个元素是相互之间的距离.
          (-distance) 传入交叉熵:
            注意看文档. nll和cross的区别, 某种意义是等价的:

            ```python
            logits = torch.FloatTensor(
                [
                    [-0.1, -0.2, -0.3, -0.4, -0.5],
                    [-0.3, -0.4, -0.5, -0.1, -0.2],
                    [-0.3, -0.2, -0.4, -0.5, -0.1],
                ]
            )
            y = torch.LongTensor(
                [4, 3, 2]
            )
            nllloss = torch.nn.NLLLoss()
            nncross = nn.CrossEntropyLoss()
            fcross = F.cross_entropy

            print(nllloss(nn.LogSoftmax(dim=1)(logits), y))
            print(nncross(logits, y))
            print(fcross(logits, y))
            ```
```

+ conv_block:
  
  ```python
  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
  ```

  如上给进去的参数是初始化的, forward的过程经过这些layer的shape变化要看文档.

  以Conv2d为例, 经过这层后 就是卷积, shape: (N x C_{in} x H x W) -> (N x C_{out} x H_{out} x W_{out})

  其中 C_{in} C_{out} 就是初始化需要传进去的参数. 那个3是kernel_size, 就是filter.

  H_{out} x W_{out} 根据卷积原理可算, 文档里也有公式.

  ```python
  shape: (80 x 3 x 84 x 84) 
    --Conv2d-> (80 x 64 x H_{out} x W_{out})  # 这里64是Conv2d的C_{out}.
      --BatchNorm2d-> (80 x 64 x H_{out} x W_{out})
        --ReLU-> (80 x 64 x H_{out} x W_{out})
          --MaxPool2d-> (80 x 64 x H_{out_1} x W_{out_1})
  ```
  
+ Docs > torch.nn > CrossEntropyLoss
  
  为什么传进 交叉熵 的是 -distance?

    解: 
  
  交叉熵 和 KL散度(相对熵)区别?

    解: KL散度第一项就是交叉熵. 第二项仅与y的分布有关, 所以最小化谁都一样.


```python
MAML:
核心: 有一个非常重要的函数: functional_forward
  作用就是, 把网络的参数weight取出来, 
  然后 functional_forward(weight) 这样让x通过这些weight进行forward

1-shot 5-way 5-query
`N` = shot * way + query * way = 5 + 25 = 30

for epoch:
  for batch:
    (meta_batch_size x N x C x H x W)
    for meta_batch:
      (N x C x H x W)
      for inner_batch 次循环:
        forward一次, 算loss.
        算梯度.
        手动更新网络参数.
      (这里网络的参数有 conv`x` block 的每一层, 还有个logits)

      forward一次, 算loss.
      加入task预测值.
      算梯度, 记录梯度. 这里没有更新网络参数

      order = 1:
        ... 中间未懂, 但是有更新模型.
        optimiser.step()

二、
for epoch:
  for batch:
    for task_num:
      1 对support: forward, 算loss, 正常算梯度.
      2 更新一步网络参数到 fast_weights, 但是此时真正网络内的参数 net.parameters() 没更新.
      3 with torch.no_grad():
          1 对query: forward, 算没有更新的网络参数的loss. (基于 net.parameters())
          2 对query: forward, 算更新了的网络参数的loss.   (基于 fast_weight)
      for update_steps 次循环:
        1 对support: forward, 用fast_weight. 并算loss.
        2 算梯度.
        3 更新 fast_weight.
        4 对query: forward, 用fast_weight. 并算loss. 不更新fast_weight.


```

