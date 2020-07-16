import numpy as np
import matplotlib.pyplot as plt
import math
import random
# from scipy.special import KDTree
from sklearn import datasets
from Visitlist import visitlist
from scipy.spatial import cKDTree as KDTree
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# 计算欧几里得距离
def dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.power(a-b, 2).sum())

def dbscan_2(dataSet, eps, minPts) :

    # nPoints = dataSet.shape[0]
    nPoints = len(dataSet)

    vPoint = visitlist(count = nPoints)
    # 初始化簇标记列表C，簇标记为 k
    k = -1
    count = 0
    count_noise = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    C = [ -1 for i in range(nPoints)]

    #构建KD-Tree ,并生成所有距离<=eps的点集合
   # kd = KDTree(dataSet)

    while(vPoint.unvisitedlistNum > 0):
        # randomly choose a point called p which is unvisited
        p = random.choice(vPoint.unvisitedlist)
        vPoint.visit(p)

       # N = kd.query_ball_point(dataSet[p], eps)
        N = [i for i in range(nPoints) if dist(dataSet[i], dataSet[p]) <= eps]
        if len(N) >= minPts:
            # (6) 创建个一个新簇C，并把p添加到C
            # 这里的C是一个标记列表，直接对第p个结点进行赋值
            k += 1
            count += 1
            C[p] = k
            # (7) 令N为p的$\varepsilon$-邻域中的对象的集合
            # N是p的$\varepsilon$-邻域点集合
            # (8) for N中的每个点p'
            for p1 in N:
                # (9) if p'是unvisited
                if p1 in vPoint.unvisitedlist:
                    # (10) 标记p'为visited
                    vPoint.visit(p1)
                    # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                    # 找出p'的$\varepsilon$-邻域点，并将这些点去重新添加到N
                    # M = kd.query_ball_point(dataSet[p1], eps)
                    M = [i for i in range(nPoints) if dist(dataSet[i], dataSet[p1]) <= eps]
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    # (12) if p'还不是任何簇的成员，把p'添加到c
                    # C是标记列表，直接把p'分到对应的簇里即可
                    if C[p1] == -1 :
                        C[p1] = k
                        if k == 0 :
                            count_1 += 1
                        if k == 1 :
                            count_2 += 1
                        if k == 2 :
                            count_3 += 1
        # (15) else标记p为噪声
        else:
            C[p] = -1
            count_noise += 1

    # (16) until没有标记为unvisited的对象
    print("分类数：" + str(count))
    print("噪声数：" + str(count_noise))
    print("count_1：" + str(count_1))
    print("count_2：" + str(count_2))
    print("count_3：" + str(count_3))
    return C

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('	')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

# 构造数据集
# X1, y1 = datasets.make_circles(n_samples=2000, factor=.6, noise=.05)
# X2, Y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[-1.2, 1.5],[-1.5, -1],[0,0]], cluster_std=[0.2,0.15,0.1], random_state=1)
# X = np.concatenate((X1, X2))
# seedDataSets = mat(loadDataSet('seeds_dataset.txt'))

file=open('seeds_dataset1.txt')
dataMat=[]
labelMat=[]
for line in file.readlines():

    curLine = line.strip().split()
    # python3下的map()
    # 函数返回类型为iterators，不再是list
    # 这里使用的是map函数直接把数据转化成为float类型
    floatLine=list(map(float,curLine))
    dataMat.append(floatLine[0:6])
    # 最后一个为label标签
    labelMat.append(floatLine[-1])

#数据集的数据个数
m = len(dataMat)
print("data is ready！")
begin = time.time()
eps = 0.98
minPts = 10
result = dbscan_2(dataMat, eps, minPts)
end = time.time()
duration  = end - begin
print("my dbscan用时:" + str(duration) )

plt.figure(figsize = (12, 9), dpi=80)
dataMat = np.array(dataMat)
# 载入3D模块
fig = plt.figure()
ax = Axes3D(fig)
# 画出三维图像
ax.scatter(dataMat[:,1], dataMat[:,3],dataMat[:,5], c=result)

plt.show()

