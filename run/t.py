# import numpy as np
# from sklearn.decomposition import PCA
# import torch

# X = np.array([[-1, -1, 3], [-2, -1, 8], [7, -2, 1], [1, 1, 5], [2, 1, 9], [3, 6, 2]])
# X_t = torch.from_numpy(X).double().unsqueeze(0)

# from models.utils import get_transform_matrices_handle


# # covv = np.cov(X.T, rowvar=True)
# # U,S,V = np.linalg.svd(covv)
# # print("lailai")
# # print(U, S, V)
# # val, vec = np.linalg.eig(covv)
# # print(val, vec)

# # print(X_t @ get_transform_matrix(X_t, torch.mean(X_t, 1, keepdim=True), 'pca_pure'))

# # print(get_transform_matrix(X_t, torch.mean(X_t, 1, keepdim=True), 'pca_pure'))

# # result is a (5,5) matrix of correlations between rows
# np_corr = np.corrcoef(X.T)
# funcc = get_transform_matrices_handle(X_t, torch.mean(X_t, 1, keepdim=True))

# import torch

# y = [0, 0, 0, 1, 1, 1, 2, 0, 1, 1]
# x = torch.rand(10, 3)

# print(x)

# import pickle
# import torch
# with open('/home/lus/zhangyk/pre-trained_weights/miniImagenet/base_features.plk', 'rb') as f:
#     data = pickle.load(f)
#     for i, v in data.items():
#         data[i] = torch.FloatTensor(v).to(torch.device('cuda'))

# with open('/home/lus/zhangyk/pre-trained_weights/wrn_base_features.plk', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# from sklearn.preprocessing import StandardScaler
# data = [[0.2, 5], [0.6, 4.3], [3.1, 1], [3, 1]]
# scaler = StandardScaler()
# print(scaler.fit(data))

# print(scaler.transform(data))
# print(scaler.mean_)
# print(scaler.scale_)
# import numpy as np
# print(np.nanstd(data, axis=0))

# import scipy.stats as stats
# print(stats.zscore(data, axis=0))


# import torch
# t = torch.FloatTensor(data)
# print(torch.mean(t, dim=0, keepdim=True))
# print(torch.std(t, dim=0, unbiased=False, keepdim=True))
# z_score = (t - torch.mean(t, dim=0, keepdim=True)) / torch.std(t, dim=0, unbiased=False, keepdim=True)
# print(z_score)

# import torch
# t = torch.randn(3, 5) + 10
# t[0, 3] = 5
# print('t', t)


# def torch_cov(x, cur_mean):
#     c = x - cur_mean
#     return (1.0 / (x.shape[0] - 1)) * (torch.transpose(c, -2, -1) @ c)


# def cov(m, rowvar=False):
#     '''Estimate a covariance matrix given data.

#     Covariance indicates the level to which two variables vary together.
#     If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
#     then the covariance matrix element `C_{ij}` is the covariance of
#     `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

#     Args:
#         m: A 1-D or 2-D array containing multiple variables and observations.
#             Each row of `m` represents a variable, and each column a single
#             observation of all those variables.
#         rowvar: If `rowvar` is True, then each row represents a
#             variable, with observations in the columns. Otherwise, the
#             relationship is transposed: each column represents a variable,
#             while the rows contain observations.

#     Returns:
#         The covariance matrix of the variables.
#     '''
#     if m.dim() > 2:
#         raise ValueError('m has more than 2 dimensions')
#     if m.dim() < 2:
#         m = m.view(1, -1)
#     if not rowvar and m.size(0) != 1:
#         m = m.t()
#     # m = m.type(torch.double)  # uncomment this line if desired
#     fact = 1.0 / (m.size(1) - 1)
#     m -= torch.mean(m, dim=1, keepdim=True)
#     mt = m.t()  # if complex: mt = m.t().conj()
#     # print(m.matmul(mt))
#     return fact * m.matmul(mt).squeeze()

# # print(torch_cov(t, torch.mean(t, dim=1, keepdim=True)))
# # torch_cov(t, torch.mean(t, dim=1, keepdim=True))
# # cov(t, False)
# import numpy as np
# print('my', torch_cov(t, torch.mean(t, dim=0, keepdim=True)))
# print('np', np.cov(t.cpu().detach().numpy().T))
# print(t)
# assert 0
# print(cov(t, False))
# assert 0
# print(t)
# print(torch.mean(t, dim=0, keepdim=True).shape)
# print(torch_cov(t, torch.mean(t, dim=1, keepdim=True)))
# print(np.cov(t.cpu().detach().numpy()))



# import os
# def gpu_state(gpu_id):
#     qargs = ['index', 'gpu_name', 'memory.used', 'memory.total']
#     cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))

#     results = os.popen(cmd).readlines()
#     gpu_id_list = gpu_id.split(",")
#     for cur_state in results:
#         cur_state = cur_state.strip().split(", ")
#         for i in gpu_id_list:
#             if i == cur_state[0]:
#                 print(f'GPU {i} {cur_state[1]}: Memory-Usage {cur_state[2]} / {cur_state[3]}.')

# gpu_state('2')


# t = ['frontend3D.0.weight', 'frontend3D.1.weight', 'frontend3D.1.bias', 'frontend3D.1.running_mean', 'frontend3D.1.running_var', 'frontend3D.1.num_batches_tracked', 'resnet18.conv1.weight', 'resnet18.bn1.weight', 'resnet18.bn1.bias', 'resnet18.bn1.running_mean', 'resnet18.bn1.running_var', 'resnet18.bn1.num_batches_tracked', 'resnet18.layer1.0.conv1.weight', 'resnet18.layer1.0.bn1.weight', 'resnet18.layer1.0.bn1.bias', 'resnet18.layer1.0.bn1.running_mean', 'resnet18.layer1.0.bn1.running_var', 'resnet18.layer1.0.bn1.num_batches_tracked', 'resnet18.layer1.0.conv2.weight', 'resnet18.layer1.0.bn2.weight', 'resnet18.layer1.0.bn2.bias', 'resnet18.layer1.0.bn2.running_mean', 'resnet18.layer1.0.bn2.running_var', 'resnet18.layer1.0.bn2.num_batches_tracked', 'resnet18.layer1.1.conv1.weight', 'resnet18.layer1.1.bn1.weight', 'resnet18.layer1.1.bn1.bias', 'resnet18.layer1.1.bn1.running_mean', 'resnet18.layer1.1.bn1.running_var', 'resnet18.layer1.1.bn1.num_batches_tracked', 'resnet18.layer1.1.conv2.weight', 'resnet18.layer1.1.bn2.weight', 'resnet18.layer1.1.bn2.bias', 'resnet18.layer1.1.bn2.running_mean', 'resnet18.layer1.1.bn2.running_var', 'resnet18.layer1.1.bn2.num_batches_tracked', 'resnet18.layer2.0.conv1.weight', 'resnet18.layer2.0.bn1.weight', 'resnet18.layer2.0.bn1.bias', 'resnet18.layer2.0.bn1.running_mean', 'resnet18.layer2.0.bn1.running_var', 'resnet18.layer2.0.bn1.num_batches_tracked', 'resnet18.layer2.0.conv2.weight', 'resnet18.layer2.0.bn2.weight', 'resnet18.layer2.0.bn2.bias', 'resnet18.layer2.0.bn2.running_mean', 'resnet18.layer2.0.bn2.running_var', 'resnet18.layer2.0.bn2.num_batches_tracked', 'resnet18.layer2.0.downsample.0.weight', 'resnet18.layer2.0.downsample.1.weight', 'resnet18.layer2.0.downsample.1.bias', 'resnet18.layer2.0.downsample.1.running_mean', 'resnet18.layer2.0.downsample.1.running_var', 'resnet18.layer2.0.downsample.1.num_batches_tracked', 'resnet18.layer2.1.conv1.weight', 'resnet18.layer2.1.bn1.weight', 'resnet18.layer2.1.bn1.bias', 'resnet18.layer2.1.bn1.running_mean', 'resnet18.layer2.1.bn1.running_var', 'resnet18.layer2.1.bn1.num_batches_tracked', 'resnet18.layer2.1.conv2.weight', 'resnet18.layer2.1.bn2.weight', 'resnet18.layer2.1.bn2.bias', 'resnet18.layer2.1.bn2.running_mean', 'resnet18.layer2.1.bn2.running_var', 'resnet18.layer2.1.bn2.num_batches_tracked', 'resnet18.layer3.0.conv1.weight', 'resnet18.layer3.0.bn1.weight', 'resnet18.layer3.0.bn1.bias', 'resnet18.layer3.0.bn1.running_mean', 'resnet18.layer3.0.bn1.running_var', 'resnet18.layer3.0.bn1.num_batches_tracked', 'resnet18.layer3.0.conv2.weight', 'resnet18.layer3.0.bn2.weight', 'resnet18.layer3.0.bn2.bias', 'resnet18.layer3.0.bn2.running_mean', 'resnet18.layer3.0.bn2.running_var', 'resnet18.layer3.0.bn2.num_batches_tracked', 'resnet18.layer3.0.downsample.0.weight', 'resnet18.layer3.0.downsample.1.weight', 'resnet18.layer3.0.downsample.1.bias', 'resnet18.layer3.0.downsample.1.running_mean', 'resnet18.layer3.0.downsample.1.running_var', 'resnet18.layer3.0.downsample.1.num_batches_tracked', 'resnet18.layer3.1.conv1.weight', 'resnet18.layer3.1.bn1.weight', 'resnet18.layer3.1.bn1.bias', 'resnet18.layer3.1.bn1.running_mean', 'resnet18.layer3.1.bn1.running_var', 'resnet18.layer3.1.bn1.num_batches_tracked', 'resnet18.layer3.1.conv2.weight', 'resnet18.layer3.1.bn2.weight', 'resnet18.layer3.1.bn2.bias', 'resnet18.layer3.1.bn2.running_mean', 'resnet18.layer3.1.bn2.running_var', 'resnet18.layer3.1.bn2.num_batches_tracked', 'resnet18.layer4.0.conv1.weight', 'resnet18.layer4.0.bn1.weight', 'resnet18.layer4.0.bn1.bias', 'resnet18.layer4.0.bn1.running_mean', 'resnet18.layer4.0.bn1.running_var', 'resnet18.layer4.0.bn1.num_batches_tracked', 'resnet18.layer4.0.conv2.weight', 'resnet18.layer4.0.bn2.weight', 'resnet18.layer4.0.bn2.bias', 'resnet18.layer4.0.bn2.running_mean', 'resnet18.layer4.0.bn2.running_var', 'resnet18.layer4.0.bn2.num_batches_tracked', 'resnet18.layer4.0.downsample.0.weight', 'resnet18.layer4.0.downsample.1.weight', 'resnet18.layer4.0.downsample.1.bias', 'resnet18.layer4.0.downsample.1.running_mean', 'resnet18.layer4.0.downsample.1.running_var', 'resnet18.layer4.0.downsample.1.num_batches_tracked', 'resnet18.layer4.1.conv1.weight', 'resnet18.layer4.1.bn1.weight', 'resnet18.layer4.1.bn1.bias', 'resnet18.layer4.1.bn1.running_mean', 'resnet18.layer4.1.bn1.running_var', 'resnet18.layer4.1.bn1.num_batches_tracked', 'resnet18.layer4.1.conv2.weight', 'resnet18.layer4.1.bn2.weight', 'resnet18.layer4.1.bn2.bias', 'resnet18.layer4.1.bn2.running_mean', 'resnet18.layer4.1.bn2.running_var', 'resnet18.layer4.1.bn2.num_batches_tracked', 'resnet18.bn2.weight', 'resnet18.bn2.bias', 'resnet18.bn2.running_mean', 'resnet18.bn2.running_var', 'resnet18.bn2.num_batches_tracked', 'gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0', 'gru.weight_ih_l0_reverse', 'gru.weight_hh_l0_reverse', 'gru.bias_ih_l0_reverse', 'gru.bias_hh_l0_reverse', 'gru.weight_ih_l1', 'gru.weight_hh_l1', 'gru.bias_ih_l1', 'gru.bias_hh_l1', 'gru.weight_ih_l1_reverse', 'gru.weight_hh_l1_reverse', 'gru.bias_ih_l1_reverse', 'gru.bias_hh_l1_reverse', 'gru.weight_ih_l2', 'gru.weight_hh_l2', 'gru.bias_ih_l2', 'gru.bias_hh_l2', 'gru.weight_ih_l2_reverse', 'gru.weight_hh_l2_reverse', 'gru.bias_ih_l2_reverse', 'gru.bias_hh_l2_reverse']
# ref = ['video_cnn.frontend3D.0.weight', 'video_cnn.frontend3D.1.weight', 'video_cnn.frontend3D.1.bias', 'video_cnn.frontend3D.1.running_mean', 'video_cnn.frontend3D.1.running_var', 'video_cnn.frontend3D.1.num_batches_tracked', 'video_cnn.resnet18.layer1.0.conv1.weight', 'video_cnn.resnet18.layer1.0.bn1.weight', 'video_cnn.resnet18.layer1.0.bn1.bias', 'video_cnn.resnet18.layer1.0.bn1.running_mean', 'video_cnn.resnet18.layer1.0.bn1.running_var', 'video_cnn.resnet18.layer1.0.bn1.num_batches_tracked', 'video_cnn.resnet18.layer1.0.conv2.weight', 'video_cnn.resnet18.layer1.0.bn2.weight', 'video_cnn.resnet18.layer1.0.bn2.bias', 'video_cnn.resnet18.layer1.0.bn2.running_mean', 'video_cnn.resnet18.layer1.0.bn2.running_var', 'video_cnn.resnet18.layer1.0.bn2.num_batches_tracked', 'video_cnn.resnet18.layer1.1.conv1.weight', 'video_cnn.resnet18.layer1.1.bn1.weight', 'video_cnn.resnet18.layer1.1.bn1.bias', 'video_cnn.resnet18.layer1.1.bn1.running_mean', 'video_cnn.resnet18.layer1.1.bn1.running_var', 'video_cnn.resnet18.layer1.1.bn1.num_batches_tracked', 'video_cnn.resnet18.layer1.1.conv2.weight', 'video_cnn.resnet18.layer1.1.bn2.weight', 'video_cnn.resnet18.layer1.1.bn2.bias', 'video_cnn.resnet18.layer1.1.bn2.running_mean', 'video_cnn.resnet18.layer1.1.bn2.running_var', 'video_cnn.resnet18.layer1.1.bn2.num_batches_tracked', 'video_cnn.resnet18.layer2.0.conv1.weight', 'video_cnn.resnet18.layer2.0.bn1.weight', 'video_cnn.resnet18.layer2.0.bn1.bias', 'video_cnn.resnet18.layer2.0.bn1.running_mean', 'video_cnn.resnet18.layer2.0.bn1.running_var', 'video_cnn.resnet18.layer2.0.bn1.num_batches_tracked', 'video_cnn.resnet18.layer2.0.conv2.weight', 'video_cnn.resnet18.layer2.0.bn2.weight', 'video_cnn.resnet18.layer2.0.bn2.bias', 'video_cnn.resnet18.layer2.0.bn2.running_mean', 'video_cnn.resnet18.layer2.0.bn2.running_var', 'video_cnn.resnet18.layer2.0.bn2.num_batches_tracked', 'video_cnn.resnet18.layer2.0.downsample.0.weight', 'video_cnn.resnet18.layer2.0.downsample.1.weight', 'video_cnn.resnet18.layer2.0.downsample.1.bias', 'video_cnn.resnet18.layer2.0.downsample.1.running_mean', 'video_cnn.resnet18.layer2.0.downsample.1.running_var', 'video_cnn.resnet18.layer2.0.downsample.1.num_batches_tracked', 'video_cnn.resnet18.layer2.1.conv1.weight', 'video_cnn.resnet18.layer2.1.bn1.weight', 'video_cnn.resnet18.layer2.1.bn1.bias', 'video_cnn.resnet18.layer2.1.bn1.running_mean', 'video_cnn.resnet18.layer2.1.bn1.running_var', 'video_cnn.resnet18.layer2.1.bn1.num_batches_tracked', 'video_cnn.resnet18.layer2.1.conv2.weight', 'video_cnn.resnet18.layer2.1.bn2.weight', 'video_cnn.resnet18.layer2.1.bn2.bias', 'video_cnn.resnet18.layer2.1.bn2.running_mean', 'video_cnn.resnet18.layer2.1.bn2.running_var', 'video_cnn.resnet18.layer2.1.bn2.num_batches_tracked', 'video_cnn.resnet18.layer3.0.conv1.weight', 'video_cnn.resnet18.layer3.0.bn1.weight', 'video_cnn.resnet18.layer3.0.bn1.bias', 'video_cnn.resnet18.layer3.0.bn1.running_mean', 'video_cnn.resnet18.layer3.0.bn1.running_var', 'video_cnn.resnet18.layer3.0.bn1.num_batches_tracked', 'video_cnn.resnet18.layer3.0.conv2.weight', 'video_cnn.resnet18.layer3.0.bn2.weight', 'video_cnn.resnet18.layer3.0.bn2.bias', 'video_cnn.resnet18.layer3.0.bn2.running_mean', 'video_cnn.resnet18.layer3.0.bn2.running_var', 'video_cnn.resnet18.layer3.0.bn2.num_batches_tracked', 'video_cnn.resnet18.layer3.0.downsample.0.weight', 'video_cnn.resnet18.layer3.0.downsample.1.weight', 'video_cnn.resnet18.layer3.0.downsample.1.bias', 'video_cnn.resnet18.layer3.0.downsample.1.running_mean', 'video_cnn.resnet18.layer3.0.downsample.1.running_var', 'video_cnn.resnet18.layer3.0.downsample.1.num_batches_tracked', 'video_cnn.resnet18.layer3.1.conv1.weight', 'video_cnn.resnet18.layer3.1.bn1.weight', 'video_cnn.resnet18.layer3.1.bn1.bias', 'video_cnn.resnet18.layer3.1.bn1.running_mean', 'video_cnn.resnet18.layer3.1.bn1.running_var', 'video_cnn.resnet18.layer3.1.bn1.num_batches_tracked', 'video_cnn.resnet18.layer3.1.conv2.weight', 'video_cnn.resnet18.layer3.1.bn2.weight', 'video_cnn.resnet18.layer3.1.bn2.bias', 'video_cnn.resnet18.layer3.1.bn2.running_mean', 'video_cnn.resnet18.layer3.1.bn2.running_var', 'video_cnn.resnet18.layer3.1.bn2.num_batches_tracked', 'video_cnn.resnet18.layer4.0.conv1.weight', 'video_cnn.resnet18.layer4.0.bn1.weight', 'video_cnn.resnet18.layer4.0.bn1.bias', 'video_cnn.resnet18.layer4.0.bn1.running_mean', 'video_cnn.resnet18.layer4.0.bn1.running_var', 'video_cnn.resnet18.layer4.0.bn1.num_batches_tracked', 'video_cnn.resnet18.layer4.0.conv2.weight', 'video_cnn.resnet18.layer4.0.bn2.weight', 'video_cnn.resnet18.layer4.0.bn2.bias', 'video_cnn.resnet18.layer4.0.bn2.running_mean', 'video_cnn.resnet18.layer4.0.bn2.running_var', 'video_cnn.resnet18.layer4.0.bn2.num_batches_tracked', 'video_cnn.resnet18.layer4.0.downsample.0.weight', 'video_cnn.resnet18.layer4.0.downsample.1.weight', 'video_cnn.resnet18.layer4.0.downsample.1.bias', 'video_cnn.resnet18.layer4.0.downsample.1.running_mean', 'video_cnn.resnet18.layer4.0.downsample.1.running_var', 'video_cnn.resnet18.layer4.0.downsample.1.num_batches_tracked', 'video_cnn.resnet18.layer4.1.conv1.weight', 'video_cnn.resnet18.layer4.1.bn1.weight', 'video_cnn.resnet18.layer4.1.bn1.bias', 'video_cnn.resnet18.layer4.1.bn1.running_mean', 'video_cnn.resnet18.layer4.1.bn1.running_var', 'video_cnn.resnet18.layer4.1.bn1.num_batches_tracked', 'video_cnn.resnet18.layer4.1.conv2.weight', 'video_cnn.resnet18.layer4.1.bn2.weight', 'video_cnn.resnet18.layer4.1.bn2.bias', 'video_cnn.resnet18.layer4.1.bn2.running_mean', 'video_cnn.resnet18.layer4.1.bn2.running_var', 'video_cnn.resnet18.layer4.1.bn2.num_batches_tracked', 'video_cnn.resnet18.bn.weight', 'video_cnn.resnet18.bn.bias', 'video_cnn.resnet18.bn.running_mean', 'video_cnn.resnet18.bn.running_var', 'video_cnn.resnet18.bn.num_batches_tracked', 'gru.weight_ih_l0', 'gru.weight_hh_l0', 'gru.bias_ih_l0', 'gru.bias_hh_l0', 'gru.weight_ih_l0_reverse', 'gru.weight_hh_l0_reverse', 'gru.bias_ih_l0_reverse', 'gru.bias_hh_l0_reverse', 'gru.weight_ih_l1', 'gru.weight_hh_l1', 'gru.bias_ih_l1', 'gru.bias_hh_l1', 'gru.weight_ih_l1_reverse', 'gru.weight_hh_l1_reverse', 'gru.bias_ih_l1_reverse', 'gru.bias_hh_l1_reverse', 'gru.weight_ih_l2', 'gru.weight_hh_l2', 'gru.bias_ih_l2', 'gru.bias_hh_l2', 'gru.weight_ih_l2_reverse', 'gru.weight_hh_l2_reverse', 'gru.bias_ih_l2_reverse', 'gru.bias_hh_l2_reverse', 'v_cls.weight', 'v_cls.bias']

# for j in ref:
#     isFind = False
#     for i in t:
#         if i in j:
#             isFind = True
#             break
#     if isFind == False:
#         print(j)

# from models.utils import load_pickle
# import torch
# logits = load_pickle('/home/lus/zhangyk/kk.plk')
# y = torch.LongTensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
#          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,
#          2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,
#          3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
#          4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
#          6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
#          7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,
#          8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,
#          9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
#         10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
#         12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13,
#         13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
#         14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15,
#         15, 15, 15, 15, 15, 15]).to(torch.device('cuda'))

# import torch.nn as nn

# nllloss = torch.nn.NLLLoss()
# nncross = nn.CrossEntropyLoss()
# print(logits)
# print(nllloss(nn.LogSoftmax(dim=1)(logits), y))
# print(nncross(logits, y))
# print(nn.LogSoftmax(dim=1)(logits).shape)

# logits = torch.FloatTensor(
#     [
#         [0.1, 9.9],
#         [0.1, 9.9],
#         [9.9, 0.1],
#     ]
# )
# y = torch.LongTensor(
#     [0, 0, 1]
# )
# nllloss = torch.nn.NLLLoss()
# nncross = nn.CrossEntropyLoss()

# print(nllloss(nn.LogSoftmax(dim=1)(logits), y))
# print(nncross(logits, y))

# class A():
#     def __init__(self):
#         self.x = 1
#     def f(self):
#         print(self.x)
#         print("A")
#     def q(self):
#         print("KKK")


# class B(A):
#     def __init__(self):
#         super().__init__()
#         self.x = 2
#     def f(self):
#         print(self.x)
#         print("B")

# class C(B):
#     def __init__(self):
#         super().__init__()
#     def h(self):
#         print(self.x)

# c = C()
# c.f()
# c.q()
# c.h()
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# # S: Symbol that shows starting of decoding input
# # E: Symbol that shows starting of decoding output
# # P: Symbol that will fill in blank sequence if current batch data size is short than time steps

# def make_batch(sentences):
#     input_batch = [[src_vocab[n] for n in sentences[0].split()]]
#     output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
#     target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
#     return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

# def get_sinusoid_encoding_table(n_position, d_model):
#     def cal_angle(position, hid_idx):
#         return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(d_model)]

#     sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#     return torch.FloatTensor(sinusoid_table)

# def get_attn_pad_mask(seq_q, seq_k):
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
#     return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

# def get_attn_subsequent_mask(seq):
#     attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1)
#     subsequent_mask = torch.from_numpy(subsequent_mask).byte()
#     return subsequent_mask

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()

#     def forward(self, Q, K, V, attn_mask):
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
#         scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)
#         return context, attn

# class MultiHeadAttention(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention, self).__init__()
#         self.W_Q = nn.Linear(d_model, d_k * n_heads)
#         self.W_K = nn.Linear(d_model, d_k * n_heads)
#         self.W_V = nn.Linear(d_model, d_v * n_heads)
#         self.linear = nn.Linear(n_heads * d_v, d_model)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, Q, K, V, attn_mask):
#         # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
#         residual, batch_size = Q, Q.size(0)
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#         print(Q.shape)
        
#         assert 0
#         q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
#         k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
#         v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

#         attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

#         # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
#         context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
#         output = self.linear(context)
#         return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, inputs):
#         residual = inputs # inputs : [batch_size, len_q, d_model]
#         output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
#         output = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output + residual)

# class EncoderLayer(nn.Module):
#     def __init__(self):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = MultiHeadAttention()
#         self.pos_ffn = PoswiseFeedForwardNet()

#     def forward(self, enc_inputs, enc_self_attn_mask):
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
#         enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
#         return enc_outputs, attn

# class DecoderLayer(nn.Module):
#     def __init__(self):
#         super(DecoderLayer, self).__init__()
#         self.dec_self_attn = MultiHeadAttention()
#         self.dec_enc_attn = MultiHeadAttention()
#         self.pos_ffn = PoswiseFeedForwardNet()

#     def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
#         dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
#         dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
#         dec_outputs = self.pos_ffn(dec_outputs)
#         return dec_outputs, dec_self_attn, dec_enc_attn

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.src_emb = nn.Embedding(src_vocab_size, d_model)
#         self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
#         self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

#     def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
#         enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1,2,0,0,0]]))
#         enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

#         enc_self_attns = []
#         for layer in self.layers:
#             enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
#             enc_self_attns.append(enc_self_attn)
#         return enc_outputs, enc_self_attns

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
#         self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
#         self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

#     def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
#         dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))
#         dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
#         dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
#         dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

#         dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

#         dec_self_attns, dec_enc_attns = [], []
#         for layer in self.layers:
#             dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
#             dec_self_attns.append(dec_self_attn)
#             dec_enc_attns.append(dec_enc_attn)
#         return dec_outputs, dec_self_attns, dec_enc_attns

# class Transformer(nn.Module):
#     def __init__(self):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
#     def forward(self, enc_inputs, dec_inputs):
#         enc_outputs, enc_self_attns = self.encoder(enc_inputs)
#         dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
#         dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
#         return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

# def showgraph(attn):
#     attn = attn[-1].squeeze(0)[0]
#     attn = attn.squeeze(0).data.numpy()
#     fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
#     ax = fig.add_subplot(1, 1, 1)
#     ax.matshow(attn, cmap='viridis')
#     ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
#     ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
#     plt.show()

# if __name__ == '__main__':
#     sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

#     # Transformer Parameters
#     # Padding Should be Zero
#     src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
#     src_vocab_size = len(src_vocab)

#     tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6, 'KK': 7}
#     number_dict = {i: w for i, w in enumerate(tgt_vocab)}
#     tgt_vocab_size = len(tgt_vocab)

#     src_len = 5 # length of source
#     tgt_len = 5 # length of target

#     d_model = 512  # Embedding Size
#     d_ff = 2048  # FeedForward dimension
#     d_k = d_v = 64  # dimension of K(=Q), V
#     n_layers = 6  # number of Encoder of Decoder Layer
#     n_heads = 8  # number of heads in Multi-Head Attention

#     model = Transformer()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     enc_inputs, dec_inputs, target_batch = make_batch(sentences)
#     print(enc_inputs, dec_inputs)
#     for epoch in range(50):
#         optimizer.zero_grad()
#         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#         loss = criterion(outputs, target_batch.contiguous().view(-1))
#         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
#         loss.backward()
#         optimizer.step()

#     # Test
#     predict, _, _, _ = model(enc_inputs, dec_inputs)

#     predict = predict.data.max(1, keepdim=True)[1]
#     print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

#     print('first head of last state enc_self_attns')
#     showgraph(enc_self_attns)

#     print('first head of last state dec_self_attns')
#     showgraph(dec_self_attns)

#     print('first head of last state dec_enc_attns')
#     showgraph(dec_enc_attns)

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# def make_batch():
#     input_batch = []
#     target_batch = []

#     words = sentence.split()
#     for i, word in enumerate(words[:-1]):
#         input = [word_dict[n] for n in words[:(i + 1)]]
#         input = input + [0] * (max_len - len(input))
#         target = word_dict[words[i + 1]]
#         input_batch.append(np.eye(n_class)[input])
#         target_batch.append(target)

#     return input_batch, target_batch

# class BiLSTM(nn.Module):
#     def __init__(self):
#         super(BiLSTM, self).__init__()

#         print(n_class, n_hidden)
#         self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
#         self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
#         self.b = nn.Parameter(torch.ones([n_class]))

#     def forward(self, X):
#         print(X.shape)
#         input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

#         hidden_state = torch.zeros(1*2, len(X), n_hidden)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
#         cell_state = torch.zeros(1*2, len(X), n_hidden)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

#         print(len(X))
#         print(hidden_state.shape) # 2 
#         print(input.shape)
#         assert 0
#         outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
#         outputs = outputs[-1]  # [batch_size, n_hidden]
#         model = self.W(outputs) + self.b  # model : [batch_size, n_class]
#         return model

# if __name__ == '__main__':
#     n_hidden = 5 # number of hidden units in one cell

#     sentence = (
#         'Lorem ipsum dolor sit amet consectetur adipisicing elit '
#         'sed do eiusmod tempor incididunt ut labore et dolore magna '
#         'aliqua Ut enim ad minim veniam quis nostrud exercitation'
#     )

#     word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
#     number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
#     n_class = len(word_dict)
#     max_len = len(sentence.split())

#     model = BiLSTM()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     input_batch, target_batch = make_batch()
#     input_batch = torch.FloatTensor(input_batch)
#     target_batch = torch.LongTensor(target_batch)

#     # Training
#     for epoch in range(10000):
#         optimizer.zero_grad()
#         output = model(input_batch)
#         loss = criterion(output, target_batch)
#         if (epoch + 1) % 1000 == 0:
#             print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

#         loss.backward()
#         optimizer.step()

#     predict = model(input_batch).data.max(1, keepdim=True)[1]
#     print(sentence)
#     print([number_dict[n.item()] for n in predict.squeeze()])

# from models.backbone.lstm import BiLSTM

# t = BiLSTM()

# import torch
# a = torch.rand(5, 29, 512)
# print(t(a).shape)

# import torch
# import torch.nn as nn

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# m = conv1x1(29, 1)
# a = torch.rand([1, 75, 5, 29])
# a = a.permute(0, 3, 1, 2)
# print(torch.cat([a, a, a], dim=1).shape)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

def plot(data, title):
    sns.set_style('dark')
    f, ax = plt.subplots()
    ax.set(ylabel='frequency')
    ax.set(xlabel='height(blue) / weight(green)')
    ax.set(title=title)
    sns.distplot(data[:, 0:1], color='blue')
    sns.distplot(data[:, 1:2], color='green')
    plt.savefig(title + '.png')

np.random.seed(42)
height = np.random.normal(loc=168, scale=5, size=1000).reshape(-1, 1)
weight = np.random.normal(loc=70, scale=10, size=1000).reshape(-1, 1)

original_data = np.concatenate((height, weight), axis=1)
plot(original_data, 'Normalization Test Original')

import torch
tmp = torch.from_numpy(original_data)
import torch.nn.functional as F
original_data = F.normalize(tmp, p=2, dim=1, eps=1e-5).cpu().detach().numpy()
plot(original_data, 'Normalization Test After')

# standard_scaler_data = preprocessing.StandardScaler().fit_transform(original_data)
# plot(standard_scaler_data, 'StandardScaler')

# min_max_scaler_data = preprocessing.MinMaxScaler().fit_transform(original_data)
# plot(min_max_scaler_data, 'MinMaxScaler')

# max_abs_scaler_data = preprocessing.MaxAbsScaler().fit_transform(original_data)
# plot(max_abs_scaler_data, 'MaxAbsScaler')

# normalizer_data = preprocessing.Normalizer().fit_transform(original_data)
# plot(normalizer_data, 'Normalizer')

# robust_scaler_data = preprocessing.RobustScaler().fit_transform(original_data)
# plot(robust_scaler_data, 'RobustScaler')