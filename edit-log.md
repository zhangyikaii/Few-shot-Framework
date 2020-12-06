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

+ `NShotTaskSampler` 类太关键了, 所以之前没看导致relation net那边可能support/query写反了.

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

+ `NShotTaskSampler` 决定了如何 generates batches of n-shot, k-way, q-query tasks.

  `--k-train 20 --n-train 1 --q-train 15` 请注意在训练时候的 `k, n, q` 是这样的.

+ `proto_net_episode` 的参数 `y` 是query set的label. 详见 `prepare_nshot_task`.

+ 纠正过来, 事实是: 训练时候 support 1-shot, query 20-shot.


## Trick

1. 野生数据预处理, FEAT里面似乎训练和测试还不一样, `RandomResizedCrop`, `ColorJitter` 之类的. 因为测试就只能按要求来?

2. `num_tasks` 之类的参数?

3. model & data: float32 or float64.


## Few-shot-Framework

+ 2020-12-4 21:35:51, 写完了Dataloader, 没测试. `MiniImageNet` 类还是要测一下, `augment` 参数没懂, 整个类和FEAT的还是不一样的.

+ `we use ADAM for ConvNet and SGD for other backbones` 为啥其他backbone就是sgd?

+ 2020-12-5 09:03:12 接下来构建`ProtoNet`

+ 注意, `prepare_nshot_task` 还要再写. 其实就是准备 `y` Label而已.

+ 期望的结果是增加模型后不用改动太多的文件, 即不同该`utils.py`文件

+ TODO: 在迭代内, progress bar 没有被更新, 是self.length那里出错吗?