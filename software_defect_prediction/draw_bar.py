# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  

def ge_list(star,step,n):
    mylist = []
    for i in range(n):
        mylist.append(star + i*step)
    return mylist

def draw(group_list,label_list,y_1 = None,y_2 = None,y_3 = None,y_4 = None,y_5=None):

    # filename = 'sample.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    if y_1 is None:
        y_1 = np.loadtxt("D:/result/precision.txt",delimiter=',')
        print("y_1",y_1)
    if y_2 is None:
        y_2 = np.loadtxt("D:/result/AUC.txt",delimiter=',')
    if y_3 is None:
        y_3 = np.loadtxt("D:/result/recall.txt",delimiter=',')
    if y_4 is None:
        y_3 = np.loadtxt("D:/result/G_means.txt",delimiter=',')
    if y_5 is None:
        y_4 = np.loadtxt("D:/result/pf.txt",delimiter=',')


    n_group = len(group_list)
    n_contrast = len(label_list)


    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    
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

    ax4.set_xticks(np.arange(n_group)*(n_contrast+2)+(n_contrast)/2+0.5)
    ax4.set_xticklabels(group_list )
    ax4.set_xlabel('数据集')
    ax4.set_ylabel('G_means')


    ax5.set_xticks(np.arange(n_group)*(n_contrast+2)+(n_contrast)/2+0.5)
    ax5.set_xticklabels(group_list )
    ax5.set_xlabel('数据集')
    ax5.set_ylabel('pf')

    for i in range(n_contrast):
        ax1.bar(ge_list(i+1,n_contrast+2,n_group), y_1[i], label=label_list[i])
        ax2.bar(ge_list(i+1,n_contrast+2,n_group), y_2[i], label=label_list[i])
        ax3.bar(ge_list(i+1,n_contrast+2,n_group), y_3[i], label=label_list[i])
        ax4.bar(ge_list(i+1,n_contrast+2,n_group), y_4[i], label=label_list[i])
        ax5.bar(ge_list(i+1,n_contrast+2,n_group), y_5[i], label=label_list[i])




    ax1.set_title(u'不同平衡方法precision对比')
    ax1.legend()

    ax2.set_title(u'不同平衡方法AUC对比')
    ax2.legend()

    ax3.set_title(u'不同平衡方法pd对比')
    ax3.legend()

    ax4.set_title(u'不同平衡方法G_means对比')
    ax4.legend()

    ax5.set_title(u'不同平衡方法pf对比')
    ax5.legend()

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()

    current_time = time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))
    filename1 = "D:/result/picture/"+current_time+"~1.png"
    fig1.savefig(filename1)
    filename2 = "D:/result/picture/"+current_time+"~2.png"
    fig2.savefig(filename2)
    filename3 = "D:/result/picture/"+current_time+"~3.png"
    fig3.savefig(filename3)
    filename4 = "D:/result/picture/"+current_time+"~4.png"
    fig4.savefig(filename4)
    filename5 = "D:/result/picture/"+current_time+"~5.png"
    fig5.savefig(filename5)


    plt.show()


# y_1 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
#             [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
#             [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
#             [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
#             [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
#             [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

# y_2 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
#             [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
#             [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
#             [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
#             [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
#             [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

# y_3 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
#             [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
#             [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
#             [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
#             [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
#             [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

# y_4 = [[0.305008658,0.318881376,0.330473963,0.506954632,0.471911663],
#             [0.211656638,0.278524045,0.293345471,0.503062078,0.448014742],
#             [0.206507326,0.278128926,0.280914835,0.532234651,0.438566096],
#             [0.214740041,0.314745026,0.269481932,0.509809086,0.472312179],
#             [0.208448607,0.27490149,0.32042192,0.510677739,0.441637681],
#             [0.228317046,0.30085534,0.310832269,0.50360312,0.453364027]]

# group_list = ["CM1","JM1","PC1","PC4","PC5"]
# label_list = ['NONE','MAHAKIL','SMOTE','KNN_SMOTE',"borderline1","MA_RA"]

# draw(group_list,label_list,y_1,y_2,y_3,y_4)