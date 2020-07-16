import numpy as np
import matplotlib.pyplot as plt
import math
import random
# import visitlist as visitlist
from Visitlist import visitlist
from sklearn import datasets
import time

# class visitlist:
#     def __init__(self, count = 0):
#         self.unvisitedlist = [i for i in range(count)]
#         self.visitedlist = list()
#         self.unvisitedlistNum = count
#
#     def visit(self, pointID):
#         self.visitedlist.append(pointID)
#         self.unvisitedlist.remove(pointID)
#         self.unvisitedlistNum -= 1

def dist(a,b):
    # 计算a,b 两个元祖的欧几里得距离
    return math.sqrt(np.power(a-b, 2).sum())

def dbscan_1(dataSet, eps, minPts):
    # numpy.ndarray 的shape属性表示矩阵的行数与列数
    # 取行数
    nPoints = dataSet.shape[0]
    # (1)标记所有对象为unvisited
    vPoint = visitlist(count = nPoints)

    # (2)初始化簇标记列表 C，簇标记为 k
    k = -1
    C = [-1 for i in range(nPoints)]
    while(vPoint.unvisitedlistNum > 0):
        # (3)随机上选择一个unvisited对象p
        p = random.choice(vPoint.unvisitedlist)
        # (4)标记p为visited
        vPoint.visit(p)
        # (5)检查领域内是否满足minPts个点
        neighborEps = [ i for i in range(nPoints) if dist(dataSet[i], dataSet[p] ) <= eps]
        if len(neighborEps) >= minPts:
            # (6) create a new cluster C ,and put points p into C
            # C 代表着不同的簇，用k来表示簇的个数
            k += 1
            C[p] = k
            # (7) 令 neighborEps 为p的eps-领域对象的集合
            for p1 in neighborEps :
                # (9) if p1 是unvisited，则标记p1为visited
                if p1 in vPoint.unvisitedlist :
                    vPoint.visit(p1)
                    # (10) 找出p1的eps领域内有MinPts个点，把点p1加到neighborEps中
                    p1_neighborEps = [i for i in range(nPoints) if dist(dataSet[i], dataSet[p1]) <=  eps]
                    # 去重复
                    if len(p1_neighborEps) >= minPts :
                        for i in p1_neighborEps:
                            if i not in neighborEps :
                                neighborEps.append(i)

                    # (11)如果p1还不是任何簇的成员，把p1添加到C里
                    if C[p1] == -1 :
                        C[p1] = k
            # (12)否则p点就是噪声点
            else:
                C[p] = -1

    # (13)直到访问完所有点
    return C 

# 构造数据集
X1, y1 = datasets.make_circles(n_samples=15000, factor=.6, noise=.05)
X2, Y2 = datasets.make_blobs(n_samples=5000, n_features=2, centers=[[-1.2, 1.5],[-1.5, -1],[0,0]], cluster_std=[0.2,0.15,0.1], random_state=1)
X = np.concatenate((X1, X2))
print("data is ready！")
begin = time.time()
eps = 0.1
minPts = 10
result = dbscan_1(X, eps, minPts)
end = time.time()
duration  = end - begin
print("my dbscan用时:" + str(duration) )

plt.figure(figsize = (12, 9), dpi=80)
plt.scatter(X[:,0],X[:,1],c=result)
plt.show()


