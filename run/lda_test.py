SKLEARN_LDA = False
RG_LDA = True

import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import torch
sys.path.append("../")
mpl.use('Agg')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from models.few_shot.rglda import RGLDA
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

if SKLEARN_LDA:
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    y, y_idx = torch.sort(y)
    X = torch.index_select(X, 0, y_idx)
    X, y = X.detach().numpy(), y.detach().numpy()
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_ori = lda.fit(X, y).transform(X)

elif RG_LDA:
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    lda = RGLDA(X, y, solver='lap', n_components=2)
    X_lda_ori = lda.fit().transform(X)

class_idx = [0, 1, 3, 5, 6, 8, 9]
for i, target_name in zip(class_idx, [digits.target_names[i] for i in class_idx]):
    plt.scatter(X_lda_ori[y == i, 0], X_lda_ori[y == i, 1], s=15, alpha=.7, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of DIGITS dataset')
plt.savefig('lda_test_sklearn.png')


