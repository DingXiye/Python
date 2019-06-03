# -*- coding: utf-8 -*-  
# MAHAKIL 重采样
# phase 1 : 计算马氏距离，并递减排序
# phase 2 : 划分三角形
# phase 3 : 第一个盒子内选择三角形第一个顶底，第二个盒子选对应相邻的点，在三角形内选择一点作为新样本
import numpy as np


class TRI(object):
    def __init__(self, pfp=0.5):#生成与无缺陷模块1:1的缺陷样本
        self.data_t = None  # 保存初始时的缺陷样本（特征），是一个二维数组，
        #每行是一个缺陷样本的所有特征值，每列是一个特征下所有缺陷样本的对应值
        self.pfp = pfp  # 预期缺陷样本占比
        self.T = 0  # 需要生成的缺陷样本数
        self.new = []  # 存放新生成的样本

    # 核心方法
    # return : data_new, label_new
    def fit_sample(self, data, label):
        # data : 包含度量信息的样本 数组
        # label : 样本的标签 数组
        self.new = []
        data_t, data_f, label_t, label_f = [], [], [], []
        # 按照正例和反例划分数据集
        for i in range(label.shape[0]):
            if label[i] == 1:#这样每次同步加入，加入列表的顺序一致，在特征data和标签label中，序号也是一致的
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])
        self.T = len(data_f) / (1 - self.pfp) - len(data_f) -len(data_t)
        self.data_t = np.array(data_t)
        # print("训练集中缺陷样本个数：",len(data_t))
        # 计算得到马氏距离
        d = self.mahalanobis_distance(self.data_t)#这里返回的是(0,d0),(1,d1),(2,d2),....即带样本序号的马氏距离元组列表
        # 降序排序
        d.sort(key=lambda x: x[1], reverse=True)#把d按照马氏距离排序，可能会变成这样[(2,d2),(0,d0),(1,d1).....]
        # 将正例集一分为二
        k = len(d)
        d_index = [d[i][0] for i in range(k)]
        data_t_sorted = [data_t[i] for i in d_index]
        len_tr = int(k/2) #三角形的个数，即第一个盒子里个数
        bin1 = [data_t_sorted[i] for i in range(0, len_tr)]
        bin2 = [data_t_sorted[i] for i in range(len_tr, 2*len_tr)]

        self.new = self.generate_new_sample(bin1, bin2, self.T)
        # 返回数据与标签
        label_new = np.ones(len(self.new))
        return np.append(data, self.new, axis=0), np.append(label, label_new, axis=0)

    def Eucli_distance(self, X):
        n=X.shape[0]
        
        mu = np.mean(X, axis=0) 
        d1=[]
        for i in range(0,n):
                d = np.sqrt(np.sum(np.square(X[i]-mu)))
                if(d<=0):
                    print("异常",d)
                d1.append((i,d))
        return d1

    def mahalanobis_distance(self, X):
        n=X.shape[0]
        XT = X.T
        S=np.cov(XT)   #维度之间协方差矩阵,cov求得的是变量之间的协方差，默认一行为一个变量，求维度协方差要转置
        
        mu = np.mean(X, axis=0) 
        SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
        # print("SI",SI)
        d1=[]
        for i in range(0,n):
                delta=X[i]-mu
                d = np.dot(np.dot(delta,SI),delta.T)
                if(d<=0):
                    print("异常",d)
                d1.append((i,d))
        return d1


    # 生成新样本
    def generate_new_sample(self, bin1, bin2, T):
        # bin1, bin2,bin3  是数组
        # T 需要生成的新样本个数
        lv_0  = []
        while(len(lv_0)<T):
            #随机选择一个三角形，在三角形内随机选一点，作为新样本
            tri_index = int(len(lv_0)%len(bin1))
            a = np.random.rand()
            b = np.random.rand()
            if tri_index+1 < len(bin1):
                lv_0.append(a*bin1[tri_index] + b*(1-a)*bin2[tri_index] + (1-a)*(1-b)*bin2[tri_index+1])
            else:
                lv_0.append(a*bin1[tri_index] + b*(1-a)*bin2[tri_index] + (1-a)*(1-b)*bin2[tri_index-1])
            #选择和bin1中序号相同的，以及距中心距离相邻的
        
        return lv_0
