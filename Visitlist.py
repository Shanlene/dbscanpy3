# visitlist类用于记录访问列表
# unvisitedlist记录未访问过的点
# visitedlist记录已访问过的点
# unvisitednum记录访问过的点数量


class visitlist:
    def __init__(self, count = 0):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitedlistNum = count

    def visit(self, pointID):
        self.visitedlist.append(pointID)
        self.unvisitedlist.remove(pointID)
        self.unvisitedlistNum -= 1