# -*- coding: utf-8 -*-  
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold
# from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from DataSetTool import DataSetTool
from mahakil import MAHAKIL

from ROS import ROS
# from RA_TRI import RA_TRI
# from SEQ_TRI import SEQ_TRI
# from Eucli import Eucil
# from TRI import TRI
# from Replicate import Replicate
# from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
# from kmeans_smote import KMeansSMOTE
# from TRI_SMOTE import TRI_SMOTE
import draw_scatter
import draw_bar 
import os
import time



database = ['CM1.arff',"KC3.arff",'PC1.arff','PC4.arff','PC5.arff']
methods = ['NONE','MAHAKIL','SMOTE','ROS']
method_list = [None,MAHAKIL(),SMOTE(),ROS()]
data_list, label_list = DataSetTool.init_data("C:\\Users\\dingye\\Desktop\\软件缺陷\\data\\", 20, False, True,database)
database_names = [os.path.splitext(file)[0] for file in database]

#每一列是一个数据集，每一行是一种方法
precision_table = np.zeros([len(methods),len(database)])
AUC_table = np.zeros([len(methods),len(database)])
recall_table = np.zeros([len(methods),len(database)])
G_table = np.zeros([len(methods),len(database)])
pf_table=np.zeros([len(methods),len(database)])

for database_index in range (len(database)):
    data = data_list[database_index]
    label = label_list[database_index]
    # MAHAKIL().fit_sample(data, label)

    feature_A = data
    label_A = label


    n_test = len(methods)#4个对比实验
    n = 20 #20次重复实验
    performance_len = 5 #性能评估指标个数
    k = 5  #5折交叉实验
    Stratified_or_not = True

    def pr_model(train_feature,train_label,test_feature,test_label):
        clf = DecisionTreeClassifier(min_samples_leaf=1)
        # clf = GaussianNB()
        clf.fit(train_feature,train_label)
        score = clf.predict_proba(test_feature)[:, 1]
        pred = clf.predict(test_feature)
        return score,pred

    def calculate(label,score):
        result = metrics.roc_curve(y_true=label, y_score=score, pos_label=1)#假样率，真阳率，（判断正负样本的阈值，不一定0.5）阈值
        fpr = result[0]
        tpr = result[1]
        auc = metrics.auc(fpr, tpr)
        # print("最佳阈值为：",thresholds)
        return auc

    def cal_G_means(label,pred):
        TP,TN,FP,FN = cal_T_P(label,pred)
        G_means = np.sqrt(TP*TN/((TP+FN)*(TN+FP)))
        return G_means

    def cal_pf(label,pred):
        TP,TN,FP,FN = cal_T_P(label,pred)
        pf= FP/(TN+FP)
        return pf

    def cal_pd(label,pred):
        TP,TN,FP,FN = cal_T_P(label,pred)
        pd= TP/(TP+FN)
        return pd

    def cal_precision(label,pred):
        TP,TN,FP,FN = cal_T_P(label,pred)
        precision=TP/(TP+FP)
        return precision


    def cal_T_P(label,pred):
        TP,TN,FP,FN = 0,0,0,0
        for i in range(len(label)):
            if label[i] == pred[i]:
                if pred[i] ==1:
                    TP += 1
                else:
                    TN += 1
            elif pred[i] == 1:
                FP += 1
            else:
                FN += 1

        return TP,TN,FP,FN

    def write_into_txt(file_name,tabel):
        file_name = "D:/result/"+file_name
        f = open(file_name,"a")
        current_time = time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))
        f.write(current_time)
        f.write("\n")
        f.write(','.join(str(s) for s in np.mean(tabel,axis=0)))
        f.write("\n")
        f.close()
        pass

    #初始化性能指标记录列表
    perform_list = [None]*n_test
    for i in np.arange(n_test):
        perform_list[i] = np.ones([n*k,performance_len])#k折n次一起平均

    j = 0 #第j个性能结果
    for i in range(n):    
        print(database[database_index],"****************************第",i,"次重复实验*****************")
        #随机重排序
        re_index = np.random.permutation(len(label_A))
        feature_A = feature_A[re_index]
        label_A = label_A[re_index]

        if Stratified_or_not  is True:
            kf = StratifiedKFold(n_splits=k) 
            # print("分层划分K折交叉")
        else:
            kf = KFold(n_splits=k)
            # print("随机划分K折交叉")
        
        k_no = 0
        for train_index,test_index in kf.split(feature_A,label_A):
            print("****************************第",k_no,"折*****************")
            train_feature = [None]*len(method_list)
            train_label = [None]*len(method_list)
            args = [None]*len(method_list)

            root_train_feature = np.array(feature_A[train_index])
            root_train_label = label_A[train_index]
            
            train_feature[0] = root_train_feature.copy()
            train_label[0] = root_train_label.copy()
            test_feature = np.array(feature_A[test_index])
            test_label = label_A[test_index]
            # draw_scatter.draw_scatter_3D(train_feature[0],train_label[0],"仅清洗后数据")
            #仅清洗后的数据
            args[0] = (train_feature[0],train_label[0],test_feature,test_label)
            
            for meth_index in range(1,len(method_list)):
                train_feature[meth_index] = root_train_feature.copy()
                train_label[meth_index] = root_train_label.copy()
                train_feature[meth_index],train_label[meth_index] = method_list[meth_index].fit_sample(train_feature[meth_index],train_label[meth_index])
                args[meth_index] = (train_feature[meth_index],train_label[meth_index],test_feature,test_label)
                # draw_scatter.draw_scatter_3D(train_feature[meth_index],train_label[meth_index],methods[meth_index])
            
            test_list = [pr_model(*args[meth_index]) for meth_index in range(len(method_list))]

            for i_test in range(len(test_list)):
                score,pred = test_list[i_test]
                #顺序为precision,AUC,recall,G-means,pf
                performance = [cal_precision(test_label, pred),
                                calculate(test_label,score),
                                # metrics.recall_score(test_label, pred),
                                cal_pd(test_label,pred),
                                cal_G_means(test_label,pred),
                                cal_pf(test_label,pred)]
                perform_list[i_test][j] = performance
            k_no = k_no + 1 #折编号

            j = j+1 #折*重复实验次数编号
    
    #一个数据集的，一个方法的性能列表,此时每一个perform_list[i_test]的元素都是一个n*k行，3（性能）列的二维，转化为一维
    for i_test in range(len(test_list)):
        perform_list[i_test] = np.mean(perform_list[i_test] ,axis=0)
        precision_table[i_test,database_index] = perform_list[i_test][0]
        AUC_table[i_test,database_index] = perform_list[i_test][1]
        recall_table[i_test,database_index] = perform_list[i_test][2]
        G_table[i_test,database_index] = perform_list[i_test][3]
        pf_table[i_test,database_index] = perform_list[i_test][4]
    print("pf_table:",pf_table)
write_into_txt("precision.txt",precision_table)   
write_into_txt("AUC.txt",AUC_table)  
write_into_txt("recall.txt",recall_table)
write_into_txt("G_means.txt",G_table)  
write_into_txt("pf.txt",pf_table) 
draw_bar.draw(database_names,methods,precision_table,AUC_table,recall_table,G_table,pf_table)