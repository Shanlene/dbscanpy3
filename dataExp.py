import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# X1, Y1 = datasets.make_circles(n_samples=2000, shuffle=True, noise=0.05, random_state=2,
#                                factor=0.6)
# X3, Y3 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05, random_state=1,)
X1, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05,random_state=1)
X2, Y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[-1.2, 1.5],[-1.5, -1],[0,0]], cluster_std=[0.2,0.15,0.1], random_state=1)

# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
              #  random_state=9)

X = np.concatenate(( X2, X1))
plt.figure(figsize=(12, 9), dpi =80)
plt.scatter(X[:, 0], X[:, 1], marker = '.')
plt.show()
