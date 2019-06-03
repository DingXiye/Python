# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

def ge_list(star,step,n):
    mylist = []
    for i in range(n):
        mylist.append(star + i*step)
    return mylist

def draw(group_list,label_list,y_1,y_2,y_3):
    '''
    JM1----->KC1  AVG:0.6218760698390964,0.6248545018829168,0.598630605956864,0.5205751454981171
    CM1----->KC1  AVG:0.6282095172885998,0.623673399520712,0.5718247175624785,0.5220472440944882
    KC1----->JM1  AVG:0.5551866628669135,0.5570247933884297,0.5472356796808208,0.6330721003134797
    CM1----->JM1  AVG:0.5664291821031633,0.5626389284696496,0.5437161584497008,0.5424337418067826
    JM1----->CM1  AVG:0.5746323529411765,0.5746323529411765,0.514390756302521,0.6333508403361343
    KC1----->CM1  AVG:0.5733718487394958,0.5640756302521008,0.5391281512605042,0.7237920168067227
    '''




    # filename = 'sample.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    # a = np.loadtxt(filename,delimiter=',')




    n_group = len(group_list)
    n_contrast = len(label_list)

    # y= a #一列一个数据集
    # y = a.T #一列一个方法，每一个bar对应的是一种对比方法，是列，所以需要转置

    # group_list = ['CM1','JM1','PC1','PC4','PC5']
    # label_list = ['NONE','MAHAKIL','SMOTE','KNN_SMOTE',"borderline1","MA_RA"]

    fig, axes = plt.subplots(2,2)
    ax1 = axes[0][0]
    ax2 = axes[0][1]
    ax3 = axes[1][0]
    
    ax1.set_xticks(np.arange(n_group)*(n_contrast+2)+(n_contrast)/2+0.5)
    ax1.set_xticklabels(group_list )
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('precision')

    ax2.set_xticks(np.arange(n_group)*(n_contrast+2)+(n_contrast)/2+0.5)
    ax2.set_xticklabels(group_list )
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('AUC')

    ax3.set_xticks(np.arange(n_group)*(n_contrast+2)+(n_contrast)/2+0.5)
    ax3.set_xticklabels(group_list )
    ax3.set_xlabel('数据集')
    ax3.set_ylabel('recall')



    for i in range(n_contrast):
        ax1.bar(ge_list(i+1,n_contrast+2,n_group), y_1[i], label=label_list[i])
        ax2.bar(ge_list(i+1,n_contrast+2,n_group), y_2[i], label=label_list[i])
        ax3.bar(ge_list(i+1,n_contrast+2,n_group), y_3[i], label=label_list[i])




    ax1.set_title(u'不同平衡方法precision对比', FontProperties=font)
    ax1.legend()

    ax2.set_title(u'不同平衡方法AUC对比', FontProperties=font)
    ax2.legend()

    ax3.set_title(u'不同平衡方法recall对比', FontProperties=font)
    ax3.legend()

    fig.tight_layout()
    plt.show()


y_1 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
            [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
            [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
            [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
            [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
            [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

y_2 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
            [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
            [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
            [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
            [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
            [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

y_3 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
            [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
            [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
            [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
            [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
            [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

group_list = ["CM1","JM1","PC1","PC4","PC5"]
label_list = ['NONE','MAHAKIL','SMOTE','KNN_SMOTE',"borderline1","MA_RA"]

draw(group_list,label_list,y_1,y_2,y_3)