# import torch
# x = torch.FloatTensor(
#     [
#         [1, 2, 3, 4, 5],
#         [9, 8, 7, 6, 8],
#         [8, 3, 6, 1, 9]
#     ]
# )
# y = torch.FloatTensor(
#     [
#         [1, 1, 1, 1, 1],
#         [3, 3, 3, 3, 3]
#     ]
# )

# n_x = x.shape[0]
# n_y = y.shape[0]
# print(x.unsqueeze(1).shape)
# print(y.unsqueeze(0).shape)

# print(x.unsqueeze(1).expand(n_x, n_y, -1).shape)
# print(y.unsqueeze(0).expand(n_x, n_y, -1).shape)
# print(x.unsqueeze(1).expand(n_x, n_y, -1))
# print(y.unsqueeze(0).expand(n_x, n_y, -1))

# distances = (
#             x.unsqueeze(1).expand(n_x, n_y, -1) -
#             y.unsqueeze(0).expand(n_x, n_y, -1)
#     ).pow(2).sum(dim=2)
# print(distances.shape)

# import torch
# import torch.nn as nn
# m = nn.Softmax(dim=1)
# t = torch.randn(2, 3).softmax(dim=1)
# print(t.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
# 测试两个loss:
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