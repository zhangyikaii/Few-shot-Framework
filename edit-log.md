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

+ `ShotTaskSamplerForDataFrame` 类太关键了, 所以之前没看导致relation net那边可能support/query写反了.

+ 2020-12-1: 下午看test部分.



## 备注:

---

models.py 里面的encoder()就是backbone.

+ experiments/ 下的每个方法文件, 相当于控制epoch的那层循环, 只是它用了fit函数(train.py) 封装实现.
    注意这里fit函数的顺序: 
      (epoch循环) on_epoch_begin -> (batch循环) on_batch_begin
      -> 参数 prepare_batch (函数) -> 参数 fit_function (函数, 就是例如matching_net_episode, few_shot/ 下面的)
      -> on_batch_end -> on_epoch_end

+ prepare_kshot_task函数(作为prepare_batch参数传入fit)运行结束后, 返回参数 y 并不是one-hot的.
    请注意, 该函数返回的:
    x: 是support和query的, 在接下来的 xxx_net_episode 函数中会被切分.
    y: 是仅有support的.

+ CrossEntropyLoss <=> Softmax-Log-NLLLoss

+ 封装 => 减少代码, 共享同样功能的代码 => 但是也减少了某些部分的灵活性. 从这个角度上看本项目的一个方法一个py是有益的.

+ 在main文件里, 一个技巧, 期望 dataset_class 根据参数变化(指向不同的类), 通过赋值即可.

+ `ShotTaskSamplerForDataFrame` 决定了如何 generates batches of k-shot, k-way, q-query tasks.

  `--k-train 20 --n-train 1 --q-train 15` 请注意在训练时候的 `k, n, q` 是这样的.

+ `proto_net_episode` 的参数 `y` 是query set的label. 详见 `prepare_kshot_task`.

+ 纠正过来, 事实是: 训练时候 support 1-shot, query 20-shot.


## Trick

1. 野生数据预处理, FEAT里面似乎训练和测试还不一样, `RandomResizedCrop`, `ColorJitter` 之类的. 因为测试就只能按要求来?

2. `num_tasks` 之类的参数?

3. model & data: float32 or float64.

4. `fix_BN` ?

5. `reg_logits` ?

6. 加速: torch 的 `contiguous()`


## Few-shot-Framework

+ 2020-12-4 21:35:51, 写完了Dataloader, 没测试. `MiniImageNet` 类还是要测一下, `augment` 参数没懂, 整个类和FEAT的还是不一样的.

+ `we use ADAM for ConvNet and SGD for other backbones` 为啥其他backbone就是sgd?

+ 2020-12-5 09:03:12 接下来构建`ProtoNet`

+ 注意, `prepare_kshot_task` 还要再写. 其实就是准备 `y` Label而已.

+ 期望的结果是增加模型后不用改动太多的文件, 即不同该`utils.py`文件

+ TODO: 在迭代内, progress bar 没有被更新, 是self.length那里出错吗?

+ 在forward函数内出错, 报错只会报: `= self.model(x)` 出错, 不会具体某一行.

+ `ValueError: too many values to unpack (expected 2)` 可能是返回了两个参数, 实际上接收那边只有一个.

+ 在 `nn` 模型内, 随着`model.eval()`, `self.traing` 在变化.

+ TODO: 注意在 `Dataloader` 那里不要分割support和query. 确认一下FEAT是怎么做的.

+ TODO: 添加终止程序后删除log file等.

+ 复制到 `E:\Few-shot-Framework` 再用git desktop更新.

+ 模型之间不同的有 (在各自模型文件里实现的): 1. `prepare_kshot_task` 函数, 2. `forward` 函数, 3. `fit_handle` 函数, 其中前两个在 `FewShotModel` 类内有一定的继承实现.

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

+ TODO: 加入pretrain:
  
  看FEAT init_weights 参数.

  加入 multi_gpu 参数.

  TODO: 看一下Adam之类的参数.

+ 重要的TODO:
  
  Dataloader的sampler方式改成FEAT的.

  看下面的例子图, FEAT的sampler方式就是对于一个取定的class, 一起sampler出这个类下support和query, 然后再划分. 这确实是更好的.

  改成同样的sampler idx.

+ TODO:
  
  2020-12-22 18:06:12:

  logs 不要在epoch_end创建然后消亡然后传参数.

  `ctrl + c` 后捕获异常, 自动删除log文件.


## Summary

在这里理清整个过程:

注意, 这里ConvNet和其他类型backbone的区别: 4个ConvNet块出来后加了一个AvgPool2d, 所以维度1600->64.
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
prototype
例子:
# 1-shot, 5-way, 15-query.
1-shot, 3-way, 2-query.
+----------------------------+
| class | support |  query   |
|   1   |   1.3   | 1.4, 1.2 |
|   3   |   3.2   | 3.1, 3.3 |
|   6   |   6.5   | 6.4, 6.2 |
+----------------------------+
一个batch如上.
同一个class下support样本与query样本不相交.
support -> 输入得到 3 个原型.
query和3个原型算距离, 得到 (6 x 3) 距离矩阵. 这6个3维vector, 在3维上那个分量大就属于哪个类.
y = 6 维: [1, 1, 3, 3, 6, 6] (但在程序里是[1, 1, 2, 2, 3, 3], 因为cross损失传入y计算的是下标).
  与label具体是啥无关!
```


```python
MatchingNet:

AttentionLSTM:

h, c 有点像计算相似度的时候那个结果, 所以每次LSTM forward计算时作为附加的计算/返回.
  query: [batch_size, 即query大小 x embedding_dim], h, c 一致.
  总过程: 
  support [(way * shot) x feature_dim], 过 BidrectionalLSTM:
    1. torch lstm 的 inputs shape: (seq_len, batch, input_size), 所以.unsqueeze(0)
    2. 过完根据hidden_size截两半:
      Appendix A.2: g(x_i, support) = h_forward_i + h_backward_i + g'(x_i)
      这里就是返回的support了, shape不变.
    3. 同时计算 _normalized
  support query, (shape如上) 同时过 AttentionLSTM:
    1. Appendix A.1 的公式 迭代计算得到.
  至此计算得到FCE embedding后的support, query, 并且shape一致.
  support_normalized和query_normalized相乘, (计算相似度), 再过一层ReLU.
    现在是[(way * query) x (way * shot].
  softmax之后每个类内**求和** 表示属于这类的权重.
    现在是[(way * query) x (way * 1]
  再过log, 再过NLLLoss.
  


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

```python
GCN 详解:

进的数据是feature [size x feature_dim] 和 adj [size x size], 注意adj是sparse的.
就这样 进到GraphConvolution层-不是torch里写的 也是feature和adj:
  
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
        在support上fast_weight functional forward一次, 算loss. 此时有fast_weight.
        由 fast_weight 算梯度.
        更新 fast_weight.
      (这里网络的参数fast_weight有 conv`x` block 的每一层, 还有个logits)

      在query上fast_weight forward一次, 算loss.
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

+ MAML 分析:
  
  $$
  \begin{array}{l}
  \min _{\theta} \sum_{\tau_{i} \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta_{i}^{\prime}}\right) \\
  \text {where } \theta_{i}^{\prime}=\theta-\alpha \nabla_{\theta} \mathcal{L}_{\mathcal{T}_{i}}\left(f_{\theta}\right)
  \end{array}
  $$

  $\theta_{i}^{\prime}$ 是 inner loop 梯度迭代的. MAML仅使用inner最终的权值进行外循环学习.

  inner里面


+ 2021-1-17 开始:

  model 的 dtype 是double吗?

  prefetch 可以做一下.

  Matching Network 有些参数初始化是否是零?


+ pretrain 相关:

  1. 在training set上(所有类别)训练一个分类器:

  ```python
  trainset = Dataset('train', args, augment=True)
  train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
  ```

  2. 假设validation set的类别数量为$V$, validation过程用 1-shot, $V$-way; 15-query; 进行验证.
     如上 15 是命令行传入的参数. 分割的时候:

     ```python
     data_shot, data_query = data[:1 * valset.num_class], data[1 * valset.num_class:] # 16-way test
     ```

     如上 `1 *` 的意思是 1-shot.


### TODO-LIST-main

(实验/构建) 百分比.

- [x] FCE.
- [ ] ProtoNet 不同shot训练. (100/0)
- [ ] ResNet-12. (100/0)
- [x] Temperature: smooth logit. (100/0)

- [x] Pretrain. (100/50) (小陷阱: 构建时注意将pretrain模型和Few-Shot模型参数对应起来 完成加载.)
- [ ] MetaOptNet. (lus上可直接跑)
- [ ] DeepEMD. (lus上可直接跑)

- [ ] **Sufficient Episodic Training** 三个方法.

- [ ] $1^{st}$-order MAML.
- [ ] ProtoMAML [Meta Dataset, ICLR20].

- [ ] SimpleShot.
- [ ] PCA-Net, 基于SimpleShot.
- [ ] LDA-Net, 基于PCA-Net.

- [x] 数据进去 加 batch_size 维度.


| GPU    | lr     | ways  | gamma | step  |
| ------ | ------ | ----- | ----- | ----- |
| l0     | 0.0005 | 5 5   | 0.5   | 20    |
| l2     | 0.001  | 5 5   | 0.5   | 20    |
| l4     | 0.01   | 5 5   | 0.3   | 20    |
| l5     | 0.001  | 30 5  | 0.5   | 10    |
| y0     | 0.001  | 5 5   | 0.4   | 20    |
| y1     | 0.001  | 30 5  | 0.7   | 20    |
| y2     | 0.001  | 30 5  | 0.5   | 30    |
| y3     | 0.001  | 5 5   | 0.5   | 10    |

ways: training way 和 validation way.
l0: 146的第0张卡.
y0: 145的第0张卡.

+ ProtoNet Conv `lr: 0.001, ways: 5 5, gamma: 0.5, step: 20`: /mnt/data3/lus/zhangyk/models/ProtoNet/0117 13-33-46-251 ProtoNet MiniImageNet ConvNet-backbone l2 5-way 5-val-way 1-shot 15-query 5-test-way 1-test-shot 15-test-query.pth

+ ProtoNet Conv `lr: 0.001, ways: 30 5, gamma: 0.4, step: 20`: /data/zhangyk/models/ProtoNet/0118 08-18-51-633 ProtoNet MiniImageNet ConvNet-backbone l2 30-way 5-val-way 1-shot 15-query 5-test-way 1-test-shot 15-test-query.pth


+ running-log:
  l0: 5-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.
  Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0213 23-35-33-770 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth; Epoch: 200 test_ accuracy: {0.572253}.

  l1: 0213 23-55-11-188 5-1-15_train-w-s-q 5-10-15_val-w-s-q 5-10-15_test-w-s-q.
  l2: 0213 23-58-07-192 30-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.
  l3: 0214 00-00-22-220 30-1-15_train-w-s-q 5-10-15_val-w-s-q 5-10-15_test-w-s-q.
smooth:
  5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q

  l4: 0214 00-05-10-090 temperature 1: Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0214 00-05-10-090 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth; Epoch: 200 test_ accuracy: {0.467535}.

  l5: 0214 00-08-53-940 temperature 2: Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0214 00-08-53-940 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth; Epoch: 200 test_ accuracy: {0.467789}.

  z0: 0214 01-06-58-466 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q
    结果: Testing model: /data/zhangyk/models/ProtoNet/0214 01-06-58-466 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth; Epoch: 200 test_ accuracy: {0.449613}.


BN训练和测试时候的差异, 目前各种实现差异较大. 类别不平衡两种加权方法: 1. 在loss上加权 2. 用sample的方法: BN中的统计量不同. 此时BN可能会因为任务带来一些问题. BN 可以让网络层数上去, 对前向传播的时候有个scale. BN 现在有个问题: 在训练和测试时候不太相同, 思考: 为什么要去掉BN.

BN 如何归一化, 如何做统计的. BN 是和传统机器学习不太一样的.

+ SimpleShot

  train_loader, test_loader直接是batchsize 256的, 不用few-shot的sample.
  
  extract_feature:
    
    在此loader下求得所有的 out_mean(dim 1600), fc_out_mean(dim 64). 即所有样本的均值, 用作后面中心化的时候减的.
    output_dict, fc_output_dict: 类别 -> 样本list的字典, 用作sample出最近邻的训练和测试用例的 (见sample_case).

  sample_case:

    先sample val_way个类别, 再每个类里sanple shot+query个.
    无奈, 就是few-shot的sample嘛. 所以之前 直接是batchsize 256 的Dataloader就是离谱.

  metric_class_type:

    算acc: 中心化减的是一样的, norm时候是不一样的(除自己).
    然后直接就是knn算了.


```
form.pptx 的记录:
Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-11-56-762 ProtoNet MiniImageNet ConvNet-backbone l2 30-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.494724}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-16-47-620 ProtoNet MiniImageNet ConvNet-backbone l2 30-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.648091}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-19-30-012 ProtoNet MiniImageNet ConvNet-backbone l2 10-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.474515}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-21-35-769 ProtoNet MiniImageNet ConvNet-backbone l2 10-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.620049}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-22-43-275 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.451380}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-24-20-274 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.567707}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-31-40-427 ProtoNet MiniImageNet ConvNet-backbone l2 10-5-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.487976}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-33-00-022 ProtoNet MiniImageNet ConvNet-backbone l2 10-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.650739}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-35-03-261 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.464897}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 03-35-50-279 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.626315}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-31-00-770 ProtoNet MiniImageNet ConvNet-backbone l2 10-10-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.491401}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-32-38-085 ProtoNet MiniImageNet ConvNet-backbone l2 10-10-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.664247}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-34-41-693 ProtoNet MiniImageNet ConvNet-backbone l2 5-10-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.470771}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-36-18-636 ProtoNet MiniImageNet ConvNet-backbone l2 5-10-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.630756}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-27-22-207 ProtoNet MiniImageNet ConvNet-backbone l2 30-10-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.475777}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 04-29-28-811 ProtoNet MiniImageNet ConvNet-backbone l2 30-10-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.652761}.

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0219 07-32-58-538 ProtoNet MiniImageNet Res12-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200 test_ accuracy: {0.480027}.
```

Testing model: /data/zhangyk/models/ProtoNet/0223 17-54-29-791 ProtoNet MiniImageNet Res12-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
测试时终止, 记得测一下.

MatchingNet 最高:
Testing model: /mnt/data3/lus/zhangyk/models/MatchingNet/0223 17-02-19-879 MatchingNet MiniImageNet Conv4-backbone cosine 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical_accuracy': 0.3864080000000345, 'test_loss': 0.0001507906779694557}


TODO:
调参有:
5-way 1-shot OK:
Testing model: /data/zhangyk/models/ProtoNet/0224 13-27-26-541 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical_accuracy': 0.4947746666666644, 'test_loss': 1.247704251050949}

source activate zykycy
python ../main.py \
    --do_train \
    --meta_batch_size 1 \
    --data_path /home/yehj/Few-Shot/data \
    --model_save_path /data/zhangyk/models \
    --max_epoch 200 \
    --gpu 0 \
    --model_class ProtoNet \
    --distance l2 \
    --backbone_class ConvNet \
    --dataset MiniImageNet \
    --train_way 5 --val_way 5 --test_way 5 \
    --train_shot 1 --val_shot 1 --test_shot 1 \
    --train_query 15 --val_query 15 --test_query 15 \
    --logger_filename /logs \
    --temperature 64 \
    --lr 0.001 --lr_mul 10 --lr_scheduler step \
    --step_size 10 \
    --gamma 1 \
    --val_interval 1 \
    --test_interval 0 \
    --loss_fn nn-cross_entropy \
    --metrics categorical_accuracy \
    --verbose \

5-way 5-shot:
Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-24-50-715 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.6703640000000001,
 'categorical accuracy (conf)': 0.0015819282856841996,
 'test_loss': 0.8553132160395384}


Pretrain (temperature):
Testing model: /data/zhangyk/models/ProtoNet/0224 22-23-38-724 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.5109133333333333,
 'categorical accuracy (conf)': 0.002005693030561423,
 'test_loss': 1.2133070271193982}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0225 15-58-01-687 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.6733626666666667,
 'categorical accuracy (conf)': 0.0015903887661499962,
 'test_loss': 0.8634194311410188}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0227 08-53-56-214 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.645244,
 'categorical accuracy (conf)': 0.0016660653855467098,
 'test_loss': 0.8912593680858613}



+ 多个shot-way之间关系的, 8个模型的:

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-37-55-103 ProtoNet MiniImageNet ConvNet-backbone l2 5-1-15_train-w-s-q 5-1-15_val-w-s-q 5-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.4571,
 'categorical accuracy (conf)': 0.0020075179270377086,
 'test_loss': 1.3015481362044812}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-39-17-817 ProtoNet MiniImageNet ConvNet-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.6330733333333334,
 'categorical accuracy (conf)': 0.0016965348541940158,
 'test_loss': 0.9263838383376598}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-43-54-602 ProtoNet MiniImageNet ConvNet-backbone l2 5-10-15_train-w-s-q 5-10-15_val-w-s-q 5-10-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.6750306666666668,
 'categorical accuracy (conf)': 0.0016368396884399506,
 'test_loss': 0.8295237638026476}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-44-24-760 ProtoNet MiniImageNet ConvNet-backbone l2 5-20-15_train-w-s-q 5-20-15_val-w-s-q 5-20-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.7117346666666666,
 'categorical accuracy (conf)': 0.0014682288117167808,
 'test_loss': 0.7600916241586209}

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0224 16-47-18-962 ProtoNet MiniImageNet ConvNet-backbone l2 10-1-15_train-w-s-q 10-1-15_val-w-s-q 10-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.316742,
 'categorical accuracy (conf)': 0.0011500958473412283,
 'test_loss': 1.888814589345455}



Testing model: /data/zhangyk/models/ProtoNet/0225 08-34-02-573 ProtoNet MiniImageNet ConvNet-backbone l2 30-1-15_train-w-s-q 16-1-15_val-w-s-q 20-1-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.21223733333333333,
 'categorical accuracy (conf)': 0.0006344249859652405,
 'test_loss': 2.5342351004838943}


实现有:
Whitening, PCA

{'categorical accuracy': 0.44753333333333295,
 'cl2n_conf': 0.0019648150930726845,
 'cl2n_mean': 0.5096266666666668,
 'l2n_conf': 0.0019008828632410479,
 'l2n_mean': 0.48781599999999997,
 'pca_conf': 0.0006862848066561804,
 'pca_mean': 0.20064666666666667,
 'un_conf': 0.0018946978748363094,
 'un_mean': 0.43896133333333326,
 'val_loss': 1.5177963227033615,
 'zca_conf': 0.0016624172337124036,
 'zca_mean': 0.44970666666666664}


Best val_ accuracy: 0.4475.
ETA: 4.5h -> 4.5h.


实现有:
~~Whitening, PCA~~

+ Whitening:

  目的: 变换后的数据方差为 $I$.

  $\Sigma^{-\frac{1}{2}}$ 是不完整的解, 因为任意的单位正交阵乘它都是解.

  $$
  X W = X Q \Sigma^{-\frac{1}{2}} = X Q U \Lambda^{-\frac{1}{2}} U^{\top}
  $$

  ZCA 零相位成分分析.

`##########################`
TODO: GPU 利用率时高时低.

+ 我的SimpleShot, 与ProtoNet的区别:

  forward后出来的维度不一样. SimpleShot用的是1600, ProtoNet过了一层AvgPool.


+ DC:
  base_mean 就是训练集上每个类的mean.

  每一个support - base_mean, 取 norm 前k大的idx, 这些idx对应的base_mean, 和query拼在一起, 求mean得到分布的mean.

  + 1 meta-train最近类中心, 2 meta-train最近类协方差
  + 生成的高斯: <每个support样本与类中心平均, 最近类协方差平均>, 从单个高斯中采样.
  + 相较于simpleshot的改进之处: 第一点 加重support与meta-train类中心的权重. 相比与PCA, 这里考虑了更多的meta-train 不同类别. 而且 第二点 该论文中多的数据是用从高斯中采样, 而不是直接PCA到低维. 第三点 线性分类器, (改进成树?)


{'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334,
 'ycy_DC_cl2n__lr_sklearn_conf': 0.003603324423823706,
 'ycy_DC_cl2n__lr_sklearn_mean': 0.6860799999999999}
{'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334,
 'ycy_DC__lr_sklearn_conf': 0.0036531370496711455,
 'ycy_DC__lr_sklearn_mean': 0.6343133333333333}

{'sklearn_LDA__protonet_conf': 0.003754781628159545,
 'sklearn_LDA__protonet_mean': 0.62154,
 'sklearn_PCA__protonet_conf': 0.0036412576765043995,
 'sklearn_PCA__protonet_mean': 0.6739533333333334,
 'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334}

DC 改进 (protonet_aug 分类器):
{'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334,
 'ycy_DC__protonet_aug_conf': 0.002667639909326428,
 'ycy_DC__protonet_aug_mean': 0.2322066666666667,
 'ycy_DC_cl2n__protonet_aug_conf': 0.0051215497584227365,
 'ycy_DC_cl2n__protonet_aug_mean': 0.3743666666666667,
 'ycy_DC_plus__protonet_aug_conf': 0.0026575743202920128,
 'ycy_DC_plus__protonet_aug_mean': 0.2318066666666667}

PCA:
{'meta_train_pca_pure__k_nn_conf': 0.003847629756750396,
 'meta_train_pca_pure__k_nn_mean': 0.5510933333333333,
 'meta_train_pca_pure__protonet_conf': 0.003640980641135022,
 'meta_train_pca_pure__protonet_mean': 0.67394,
 'sklearn_PCA__k_nn_conf': 0.003847770322436388,
 'sklearn_PCA__k_nn_mean': 0.5511,
 'sklearn_PCA__protonet_conf': 0.0036412576765043995,
 'sklearn_PCA__protonet_mean': 0.6739533333333334,
 'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334}

SimpleShot:
{'centering__protonet_conf': 0.0036412576765043995,
 'centering__protonet_mean': 0.6739533333333334,
 'cl2n__protonet_conf': 0.0036938680324138397,
 'cl2n__protonet_mean': 0.6844333333333334,
 'l2n__protonet_conf': 0.0037035723404614507,
 'l2n__protonet_mean': 0.6857399999999999,
 'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334,
 'z_score__protonet_conf': 1.2164417350911638e-18,
 'z_score__protonet_mean': 0.20000000000000004}

{'meta_train_pca_corr_whitening__protonet_conf': 0.0029674701924194244,
 'meta_train_pca_corr_whitening__protonet_mean': 0.31006,
 'meta_train_pca_pure__protonet_conf': 0.003640980641135022,
 'meta_train_pca_pure__protonet_mean': 0.67394,
 'meta_train_pca_whitening__protonet_conf': 0.002959711396284742,
 'meta_train_pca_whitening__protonet_mean': 0.31050666666666665,
 'meta_train_zca_corr_whitening__protonet_conf': 0.0031605346382942517,
 'meta_train_zca_corr_whitening__protonet_mean': 0.3473466666666667,
 'meta_train_zca_whitening__protonet_conf': 0.0025654458559937593,
 'meta_train_zca_whitening__protonet_mean': 0.2935466666666667,
 'un__protonet_conf': 0.0036412576765043995,
 'un__protonet_mean': 0.6739533333333334}

WRN:
{'un_mean': 0.7774266666666667
 'un_conf': 0.004622627744122264
 'ycy_MG_mean': 0.7214933333333333
 'ycy_MG_conf': 0.005044847329826961}

{'un_mean': 0.7759533333333334
 'un_conf': 0.0032655630132969924
 'ycy_DC_mean': 0.7782066666666667
 'ycy_DC_conf': 0.0033113578402636516}

Res-12 Pre-trained:

Testing model: /mnt/data3/lus/zhangyk/models/ProtoNet/0307-17-09-49-359 ProtoNet MiniImageNet Res12-backbone l2 5-5-15_train-w-s-q 5-5-15_val-w-s-q 5-5-15_test-w-s-q.pth
Epoch: 200:
{'categorical accuracy': 0.805536,
 'categorical accuracy (conf)': 0.001385362704986907,
 'test_loss': 1.0693040998995305}



temperature:64      64        64      256    
lr:         0.0001  0.00005   0.00005 0.0001
step_size:  10      40        40      30
gamma:      0.5     0.5       0.8     0.7

weight_decay0.0005   0.0001    0.0001
mom:        0.6      0.6       0.9
temperature:64       64        64      
lr:         0.00005  0.00005   0.00001 0.0001
step_size:  10       10        10      30
gamma:      0.5      0.5       0.5     0.7


data augment 要加上去试试.

写一个方便的输出函数.


Lip-Reading:
图像模态: https://github.com/Fengdalu/learn-an-effective-lip-reading-model-without-pains https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
音频模态: 自己做

情感识别:
文本, 音频模态: https://github.com/jbdel/MOSEI_UMONS


MiniImageNet: 100类, 每类600个, 类别之间不相交.
LRW: 500类, 首先sample出 64 base类, 16 validation类, 20 novel类, 每类600个, 存到csv里.

0.925:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.58$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 51.38$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 51.55$_{\pm 0.21}$

0.9:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.54$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 51.24$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 51.44$_{\pm 0.21}$

0.875:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.53$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 51.18$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 51.35$_{\pm 0.21}$

0.85:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.47$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 51.06$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 51.21$_{\pm 0.21}$

0.825:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.24$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 50.68$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 50.81$_{\pm 0.21}$

0.8:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.19$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 50.55$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 50.66$_{\pm 0.21}$

0.775:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.03$_{\pm 0.20}$
cl1n__tr_pca_pure__prt:                 50.34$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 50.41$_{\pm 0.21}$

0.75:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    47.73$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 49.87$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 49.95$_{\pm 0.21}$

0.725:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    47.44$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 49.56$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 49.65$_{\pm 0.21}$

0.7:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    47.29$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 49.27$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 49.36$_{\pm 0.21}$

0.675:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    47.04$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 48.92$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 48.96$_{\pm 0.21}$

0.65:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    46.94$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 48.73$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 48.78$_{\pm 0.21}$

0.625:
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    46.58$_{\pm 0.21}$
cl1n__tr_pca_pure__prt:                 48.30$_{\pm 0.21}$
cl2n__tr_pca_pure__prt:                 48.33$_{\pm 0.21}$


每个support/query样本抽出来做PCA降维:
1. 单个support/query样本 @ 与它最近的k个类的数据 堆叠起来 做PCA的特征向量矩阵, 变换后再将这些(单个support/query)堆叠起来作为结果.

'tr_s_c_copy_ratio': 0.3
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.59$_{\pm 0.20}$
c__ori__tr_s_c_pca_pure:                48.57$_{\pm 0.20}$

'tr_s_c_copy_ratio': 0.5
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.59$_{\pm 0.20}$
c__ori__tr_s_c_pca_pure:                48.57$_{\pm 0.20}$

'tr_s_c_copy_ratio': 0.7
5-way 1-shot ConvNet:
c__tr_pca_pure__prt:                    48.59$_{\pm 0.20}$
c__ori__tr_s_c_pca_pure:                48.57$_{\pm 0.20}$

learn-an-effective-lip-reading-model-without-pains 的数据不对, 要用自己划分的数据.

lrw -> prepare_lrw.py 后变成 pkl 数据, 在经过 lrw_setting_sample.py 划分出base class和novel class.
    -> base class 上自己训练pre-trained model, 做单模态的 Using the support set and unimodal comparisons 方法.

SimpleShot部分: 
方法上的尝试:
1. `trc_`: 单个support/query样本 @ 与它最近的k个类的数据 堆叠起来 做PCA的特征向量矩阵, 变换后再将这些(单个support/query)堆叠起来作为结果.
2. `tr_s_c_`: 对于单个support样本v, v copy后加入train_feature矩阵后计算的pca矩阵 u, (v @ u) 与 (query所有 @ u) 计算距离得分 作为query到当前v的距离.
3. `tr_dc_c_`: 上述copy过程变为DC生成高斯分布.


多模态部分:
1. 确定了基本路线, 确定了多模态匹配/检索的路线, 确定了网络, 还差实验与调参: 预处理 prepare_lrw.py, base/novel class的划分.
2. base class上训练pre-trained model, 先做单模态的方法.