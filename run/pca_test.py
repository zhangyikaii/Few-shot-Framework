# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# rng = np.random.RandomState(0)
# n_samples = 50
# cov = [[3, 3],
#        [3, 4]]
# X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
# pca = PCA(n_components=2).fit(X)


# plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
# for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
#     comp = comp * var  # scale component by its variance explanation power
#     print([0, comp[0]], [0, comp[1]])
#     plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
#              color=f"C{i + 2}")
# plt.gca().set(aspect='equal',
#               title="2-dimensional dataset with principal components",
#               xlabel='first feature', ylabel='second feature')
# plt.legend()
# plt.show()



# add_num = 50
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# rng = np.random.RandomState(0)
# n_samples = 50
# cov = [[3, 3],
#        [3, 4]]
# X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
# X = np.concatenate((X, np.asarray([[-2.0, 2.0]] * add_num)))
# # print(X)
# pca = PCA(n_components=2).fit(X)


# plt.scatter(X[:, 0], X[:, 1], alpha=.3, label='samples')
# for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
#     comp = comp * var  # scale component by its variance explanation power
#     print([0, comp[0]], [0, comp[1]])
#     plt.plot([0, comp[0]], [0, comp[1]], label=f"Component {i}", linewidth=5,
#              color=f"C{i + 2}")
# plt.gca().set(aspect='equal',
#               title="2-dimensional dataset with principal components",
#               xlabel='first feature', ylabel='second feature')
# plt.legend()
# plt.show()

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-2, 5, -1], [-3, 7, -2], [-1, -1, 8], [3, 1, 1], [8, 2, 1], [3, 0, 2]])
pca = PCA(n_components=3)
pca.fit(X)

print('c', pca.components_)
print(pca.singular_values_)
print('ans', pca.transform(X))
assert 0
print(np.linalg.norm((((X - X.mean(axis=0)) @ pca.components_.T) - pca.transform(X))))

import sys
sys.path.append("../")
from models.utils import get_transform_matrices_handle
import torch

U, S, V = np.linalg.svd(X - np.mean(X, axis=0), full_matrices=False)
# U, S, V = torch.svd(torch.from_numpy(X - np.mean(X, axis=0)).float())
print(S)
print(V)

def svd_flip(u, v):
       max_abs_cols = np.argmax(np.abs(u), axis=0)
       print('a', max_abs_cols)
       signs = np.sign(u[max_abs_cols, range(u.shape[1])])
       u *= signs
       print('b', signs[:, np.newaxis])
       v *= signs[:, np.newaxis]
       return u, v

def svd_flip_torch(u, v):
       max_abs_cols = torch.argmax(torch.abs(u), dim=0)
       print('a', max_abs_cols)
       signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
       u *= signs
       print('b', signs.unsqueeze(-1))
       v *= signs.unsqueeze(-1)
       return u, v

U, V = svd_flip(U, V)
print('V', V)
print(np.linalg.norm(V - pca.components_))

U, S, V = torch.svd(torch.from_numpy(X - np.mean(X, axis=0)).float())
U, V = svd_flip_torch(U, V.t())
print('V', V)
print(np.linalg.norm((((X - X.mean(axis=0)) @ pca.components_.T) - pca.transform(X))))
print(np.linalg.norm((((X - X.mean(axis=0)) @ V.cpu().detach().numpy().T) - pca.transform(X))))
print(V)
print(pca.components_)
# U, S, V = torch.svd(torch.from_numpy(X - np.mean(X, axis=0)).float())
# print(V)
assert 0


# U, V = svd_flip(U, V)
print('V', V)
assert 0

a = get_transform_matrices_handle(torch.from_numpy(X).float(), torch.mean(torch.from_numpy(X).float(), dim=0, keepdim=True), ratio=1)
# print('ans', torch.from_numpy(X).float() @ a('pca_pure'))
tt = a('pca_pure').t()
tt[1, :] = -tt[1, :]
tt = -tt[:2]
print('u.t', tt)
print('pc', pca.components_)
print(np.linalg.norm((tt.detach().numpy() - pca.components_)))
