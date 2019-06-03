# -*- coding: utf-8 -*-  
# MAHAKIL 重采样
# phase 1 : 计算马氏距离，并递减排序
# phase 2 : 分区
# phase 3 : 合成新样本
import numpy as np


class MAHAKIL(object):
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
        d_index = [d[i][0] for i in range(k)]#返回按照马氏距离降序排序的样本序号
        data_t_sorted = [data_t[i] for i in d_index]#把缺陷样本特征也按照其对应的马氏距离降序排列，这样样本特征从0到i按顺序正好就是按马氏距离降序
        mid = int(k/2)#b1，b2长度相同，或者k是奇数，b1比b2少一个
        bin1 = [data_t_sorted[i] for i in range(0, mid)]
        bin2 = [data_t_sorted[i] for i in range(mid, k)]
        # 循环迭代生成新样本
        l_ = len(bin1)
        mark = [1, 3, 7, 15, 31, 63,127]
        p = self.T / l_ #p是需要的倍数，配合上面的mark就可以知道需要多少代
        is_full = True 
        g = mark.index([m for m in mark if m > p][0]) + 1 #mark.index(obj)找出列表中出现这一项的第一个位置
        '''
        [m for m in mark if m > p][0]返回的其实就是mark中第一个大于p的数，然后用mark.index找到索引，因为List是从0计序号，
        但是我们习惯说都是第一代开始，所以加一
        '''
        cluster = 2 ** (g - 1)  # 最后一代的子代个数
        if (self.T - mark[g-2]*l_) < cluster:#避免生成超出需要的样本，宁愿少一代
            # 说明多增加一代，还不如保持少几个的状态
            is_full = False
            g -= 1
            k = 0
        else:
            k = l_ - round((self.T - mark[g-2]*l_)/cluster) # k 最后一代每个节点需要裁剪的个数
        # print("mahakil一共繁殖了",g,"代")
        self.new = self.generate_new_sample(bin1, bin2, g, l_, k, is_full)

        # 返回数据与标签
        label_new = np.ones(len(self.new))
        return np.append(data, self.new, axis=0), np.append(label, label_new, axis=0)

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
    def generate_new_sample(self, bin1, bin2, g, l, k, is_full):
        # bin1, bin2 是数组
        # g 遗传的剩余代数
        # l bin1的item数目
        # k 最后一代每个节点需要裁剪的个数
        # is_full 是否溢出，也即最后一代算完，是超出了T，还是未满T
        new_sample = []
        assert len(bin1) <= len(bin2)
        if g >= 2 or (g == 1 and is_full is False):
            lv_0 = []  # 子代
            for i in range(l):
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
                '''np.append返回的总是一维数组'''
            new_sample.extend(lv_0)
            '''extend的参数如果是可迭代的，是把参数作为列表加入，可以一下加多个成员，
                append是当做参数当做一个整体，不会自己拆开'''
            new_sample.extend(self.generate_new_sample(lv_0, bin1, g-1, l, k, is_full)) #与父亲产生一代
            new_sample.extend(self.generate_new_sample(lv_0, bin2, g-1, l, k, is_full))#与母亲产生一代
        if g == 1 and is_full:
            lv_0 = []  # 子代
            for i in range(l):
                # 生成子代
                lv_0.append(np.mean(np.append(np.atleast_2d(bin1[i]), np.atleast_2d(bin2[i]), axis=0), axis=0))
            del lv_0[-1: (-k-1): -1]#步长为负，从后往前取值，从最后一个，取到倒数第k+1个，这些要删掉
            new_sample.extend(lv_0)
        return new_sample
