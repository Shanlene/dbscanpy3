import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.colors

# 创建Figure
fig = plt.figure()
# 用来正常显示中文标签
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

X1, y1 = datasets.make_circles(n_samples=5000, factor=.6,
                                     noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2,
                            centers=[[1.2,1.2]], cluster_std=[[.1]],random_state=9)

# 原始点的分布
ax1 = fig.add_subplot(311)
X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.title(u'原始数据分布')
plt.sca(ax1)

"""
# K-means聚类
ax2 = fig.add_subplot(312)
y_pred = KMeans(n_clusters=3, random_st
from sklearn.cluster import KMeansate=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title(u'K-means聚类')
plt.sca(ax2)
"""

# DBSCAN聚类
from sklearn.cluster import DBSCAN
ax3 = fig.add_subplot(313)
y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title(u'DBSCAN聚类')
plt.sca(ax3)

plt.show()
