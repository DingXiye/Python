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
def draw_scatter():

    fig1 = plt.figure()
    Label_Com =["原样本","生成样本"]
    colors = ["b","r"]


    S_X = np.random.random(3) #随机生成三维数组
    S_Y = np.random.random(3) #随机生成三维数组

    print(S_X)
    print(S_Y)

    for i in range(3):
        print(i,":(",S_X[i],",",S_Y[i],")")

    N_X = []
    N_Y = []

    S = np.array([(S_X[i],S_Y[i]) for i in range(len(S_X))])
    print(S)
    N = []
    for j in range(100):
        a = np.random.rand()
        b = np.random.rand()
        N.append(a*S[0] + b*(1-a)*S[1] + (1-a)*(1-b)*S[2])
        N_X = [N[i][0] for i in range(len(N))]
        N_Y = [N[i][1] for i in range(len(N))]
        # N_X.append(a*S_X[0] + b*(1-a)*S_X[1] + (1-a)*(1-b)*S_X[2])
        # N_Y.append(a*S_Y[0] + b*(1-a)*S_Y[1] + (1-a)*(1-b)*S_Y[2])

    plt.scatter(S_X, S_Y, c=colors[0], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)
    plt.plot([S_X[0],S_X[1]],[S_Y[0],S_Y[1]],color = "y")  
    plt.plot([S_X[0],S_X[2]],[S_Y[0],S_Y[2]],color = "y")  
    plt.plot([S_X[1],S_X[2]],[S_Y[1],S_Y[2]],color = "y")  

    plt.scatter(N_X, N_Y, c=colors[1], cmap='brg', s=40, alpha=0.2, marker='8', linewidth=0)  

    # plt.ylim(-0.0025,0.0025)
    # plt.xlim(0.0,0.4)

    ax = fig1.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('X')
    plt.ylabel('Y')
    str = "生成新样本示意图"
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


draw_scatter()