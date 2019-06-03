# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
def draw_scatter(feature,label,str=""):

    # feature是二维，label有2种，有缺陷，无缺陷
    estimator = PCA(n_components=2)
    feature=estimator.fit_transform(feature)

    fig1 = plt.figure(1,figsize=(6,4))

    label = label.reshape((label.shape[0],1))
    data = np.concatenate((feature,label),axis=1)

    data = pd.DataFrame(data)

    colors = ['b','r']
    Label_Com = ['正常模块','缺陷模块']#有缺陷的红色，无缺陷蓝色
    for index in range(2):
        f0 = data.loc[data[2] == index][0]#f0,f1分别为两个特征，data索引2的位置是标签
        f1 = data.loc[data[2] == index][1]
        plt.scatter(f0, f1, c=colors[index], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)  

    # plt.ylim(-0.0025,0.0025)
    # plt.xlim(0.0,0.4)

    ax = fig1.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    ax.set_title(str, FontProperties=font)
    ax.legend(labels = Label_Com)

    plt.show()


def draw_scatter_3D(feature,label,str=""):
    #feature为三维时
    estimator = PCA(n_components=3)
    feature=estimator.fit_transform(feature)
    label = label.reshape((label.shape[0],1))
    data = np.concatenate((feature,label),axis=1)

    data = pd.DataFrame(data)
    colors = ['b','r']
    Label_Com = ['正常模块','缺陷模块']#有缺陷的红色，无缺陷蓝色

    fig = plt.figure()
    ax = Axes3D(fig)
    # for index in range(2):
    #     f0 = data.loc[data[3] == index][0]#f0,f1,f2分别为三个特征，data索引3的位置是标签
    #     f1 = data.loc[data[3] == index][1]
    #     f2 = data.loc[data[3] == index][2]
    #     ax.scatter(f0, f1, f2, c=colors[index], label=Label_Com[index])
    index = 1 #只显示缺陷模块
    f0 = data.loc[data[3] == index][0]#f0,f1,f2分别为三个特征，data索引3的位置是标签
    f1 = data.loc[data[3] == index][1]
    f2 = data.loc[data[3] == index][2]
    ax.scatter(f0, f1, f2, c=colors[index], label=Label_Com[index])
    
    ax.set_xlim(0.0,0.2)
    ax.set_xlabel('component x')
    ax.set_ylabel('component y')
    ax.set_zlabel('component z')
    ax.set_title(str, FontProperties=font)

    # 绘制图例
    ax.legend(loc='best')
    
    
    # 展示
    plt.show()
