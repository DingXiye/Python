# -*- coding: utf-8 -*-  
# MAHAKIL 重采样
# phase 1 : 计算欧式距离，并递减排序
# phase 2 : 划分三角形
# phase 3 : 随机选择三角形，在三角形内选择一点作为新样本
import numpy as np
import random


class TRI_SMOTE(object):
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
        data_t, label_t,  = [], []
        # 按照正例和反例划分数据集
        for i in range(label.shape[0]):
            if label[i] == 1:#这样每次同步加入，加入列表的顺序一致，在特征data和标签label中，序号也是一致的
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                pass
        self.T = (len(data) - len(data_t))/ (1 - self.pfp) - (len(data) - len(data_t)) -len(data_t)
        self.data_t = np.array(data_t)
        # print("训练集中缺陷样本个数：",len(data_t))
        # 计算得到欧式距离
        near_list = self.near_k_index(data_t,5)

        self.new = self.generate_new_sample(data_t,near_list ,self.T)
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


    def near_k_index(self,data_t,k = 5):
        near_list = np.ones((len(data_t),k))
        for i in range(len(data_t)):
            dist_list = [] #每一个样本的dist_list需要初始化为空
            for j in range(len(data_t)):
                if i != j:
                    dist = np.sqrt(np.sum(np.square(data_t[i]-data_t[j])))
                    dist_list.append((i,j,dist))
            dist_list.sort(key=lambda x: x[2]) #按照欧式距离升序排列
            '''这里有问题，上面都是对的，下面不对'''
            dist_list = dist_list[:k]
            near_list[i] = [int(tup[1]) for tup in dist_list]
        return near_list
    
    # 生成新样本
    def generate_new_sample(self, data_T,near_list, T):
        # T 需要生成的新样本个数
        lv_0  = []
        while(len(lv_0)<T):
            A_index = int(len(lv_0)%len(data_T))
            list1 = near_list[A_index].tolist()
            list2 = [int(x) for x in list1]
            B_index,C_index = random.sample(list2,2)
            a = np.random.rand()
            b = np.random.rand()
            lv_0.append(a*data_T[A_index] + b*(1-a)*data_T[B_index] + (1-a)*(1-b)*data_T[C_index])
        
        return lv_0

