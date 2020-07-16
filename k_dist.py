import  math
import numpy as np
import random
from sklearn import datasets
import time
import matplotlib.pyplot as plt

# 计算欧几里得距离
def dist(a,b):
    return math.sqrt(np.power(a-b, 2).sum())

def k_dist(dataSet, k):
    nPoints = dataSet.shape[0]
    # 创建一个nxn的矩阵
    DistMatrix = [[0 for j in range(nPoints)] for i in range(nPoints)]
    for i in range(nPoints):
        for j in range(nPoints):
            DistMatrix[i][j] = dist(dataSet[i], dataSet[j])

    for i in range(nPoints):
        #找出第k小的数
        # 对矩阵的每一行进行排序
        DistMatrix[i].sort()

    #初始化一个数组来装最小的值
    DistArray = [0 for i in range(nPoints)]

    #把第k小的数赋值到另一数组
    for i in range(nPoints):
        DistArray[i] = DistMatrix[i][k-1]

    # 距离数组从大到小排列
    DistArray.sort(reverse=True)


    for i in range(10):
        print(DistArray[i])

    return DistArray


K = 100
# 构造数据集
X1, y1 = datasets.make_circles(n_samples=2000, factor=.6, noise=.05)
X2, Y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[-1.2, 1.5],[-1.5, -1],[0,0]], cluster_std=[0.2,0.15,0.1], random_state=1)
X = np.concatenate((X1, X2))
print("data is ready！")
begin = time.time()
kDistArray = k_dist(X, 100)
end = time.time()
duration  = end - begin
print("my k-dist用时:" + str(duration) )
# 画图
plt.title(str(K) + "-dist图")
plt.xlabel("num of points")
plt.ylabel(str(K) + "-dist")
#plt.figure(figsize = (12, 9), dpi=80)
# 所有点的个数  用作横坐标
Xaxis = list(range(0,len(X)))
plt.scatter(Xaxis, kDistArray, color='b' ,linewidths= 0.0001)
plt.show()


