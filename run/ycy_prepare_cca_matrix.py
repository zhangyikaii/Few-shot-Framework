file_a = "/home/zhangyk/pre_trained_weights/Res12_base_180_features.pkl"
save_a_x_mean = "/home/zhangyk/pre_trained_weights/Res12_180_x_mean.pkl"
save_a_x_std = "/home/zhangyk/pre_trained_weights/Res12_180_x_std.pkl"
save_a_x_rotations = "/home/zhangyk/pre_trained_weights/Res12_180_x_rotations.pkl"

file_b = "/home/zhangyk/pre_trained_weights/Res12_base_features.pkl"
save_b_y_mean = "/home/zhangyk/pre_trained_weights/Res12_0_y_mean.pkl"
save_b_y_std = "/home/zhangyk/pre_trained_weights/Res12_0_y_std.pkl"
save_b_y_rotations = "/home/zhangyk/pre_trained_weights/Res12_0_y_rotations.pkl"

import sys
sys.path.append("../")
from models.utils import load_pickle, save_pickle
data_a = load_pickle(file_a)
data_b = load_pickle(file_b)

import torch
train_feature_a = torch.cat([i for i in data_a.values()], dim=0).cpu().detach().numpy()
train_feature_b = torch.cat([i for i in data_b.values()], dim=0).cpu().detach().numpy()

from sklearn.cross_decomposition import CCA

cca = CCA(n_components=256)
cca.fit(train_feature_a, train_feature_b)

save_pickle(save_a_x_mean, torch.from_numpy(cca._x_mean))
save_pickle(save_a_x_std, torch.from_numpy(cca._x_std))
save_pickle(save_a_x_rotations, torch.from_numpy(cca._x_rotations))

save_pickle(save_b_y_mean, torch.from_numpy(cca._y_mean))
save_pickle(save_b_y_std, torch.from_numpy(cca._y_std))
save_pickle(save_b_y_rotations, torch.from_numpy(cca._y_rotations))
print("OK")