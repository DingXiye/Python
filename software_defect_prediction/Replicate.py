# -*- coding: utf-8 -*-  
# 直接随机复制过采样
import numpy as np


class Replicate(object):
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
        self.new = [] #这里置空，防止直接使用之前的的方法对象，self.new的保留上一次结果
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
        while(len(self.new)<self.T):
            index = np.random.choice(range(len(data_t))) 
            self.new.append(data_t[index])

        label_new = np.ones(len(self.new))
        result_data = np.append(data, self.new, axis=0)
        result_label = np.append(label, label_new, axis=0)
        return result_data, result_label


