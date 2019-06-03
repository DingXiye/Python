# -*- coding: utf-8 -*-
# MAHAKIL 重采样
# 随机复制已存在的少数类实例来增加少数群体实例的数量。
import numpy as np


class ROS(object):
    def __init__(self, pfp=0.5):  # 生成与无缺陷模块1:1的缺陷样本
        self.data_t = None  # 保存初始时的缺陷样本（特征），是一个二维数组，
        # 每行是一个缺陷样本的所有特征值，每列是一个特征下所有缺陷样本的对应值
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
            if label[i] == 1:  # 这样每次同步加入，加入列表的顺序一致，在特征data和标签label中，序号也是一致的
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])
        self.T = len(data_f) / (1 - self.pfp) - len(data_f) - len(data_t)
        # self.data_t = np.array(data_t)
        self.new, label_new = self.generate_new_sample(data_t, data_f, self.T)
        # 返回数据与标签
        return np.append(data, self.new, axis=0), np.append(label, label_new, axis=0)

    # 生成新样本
    def generate_new_sample(self, data_t, data_f, T):
        data_add, label_new = [], []  # 新增加的
        if(len(data_f) < len(data_t)):
            for i in range(int(T)):
                index = np.random.randint(len(data_f))
                data_add.append(data_f[index])
            label_new = np.zeros(len(data_add))
        else:
            for i in range(int(T)):
                index = np.random.randint(len(data_t))
                data_add.append(data_t[index])
            label_new = np.ones(len(data_add))
        return data_add, label_new
