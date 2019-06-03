import numpy as np
import os
from mahakil import MAHAKIL
from sklearn.feature_selection import VarianceThreshold



class DataSetTool:
    # # 08版的度量补偿
    # # Mij in Target = (Mij in Target * Mean(Mj in Source)) / Mean(Mj) in  Target
    # @staticmethod
    # def metric_compensation(source, target):
    #     # 遍历每一个度量属性
    #     for j in range(target.shape[1]):
    #         # 计算每个度量属性的均值
    #         metric_mean_source = np.mean(source[:, j])
    #         metric_mean_target = np.mean(target[:, j])
    #         # 遍历每一个样例
    #         for i in range(target.shape[0]):
    #             target[i, j] = (target[i, j] * metric_mean_source) / metric_mean_target
    #     return target

    # # 17版进行调整的度量补偿
    # # Mij in Source = (Mij in Source * Mean(Mj in Target)) / Mean(Mj) in Source
    # @staticmethod
    # def metric_compensation_adopt(source, target):
    #     # 遍历每一个度量属性
    #     for j in range(source.shape[1]):
    #         # 计算每个度量属性的均值
    #         metric_mean_source = np.mean(source[:, j])
    #         metric_mean_target = np.mean(target[:, j])
    #         # 遍历每一个样例
    #         for i in range(source.shape[0]):
    #             source[i, j] = (source[i, j] * metric_mean_target) / metric_mean_source
    #     return source

    # 读取文件夹下的所有文件，并返回处理好的数据集
    # metrics_num 度量数目（原始数据中除开标签列的列数）txt文件读取时需要
    # is_sample 是否重采样
    # is_normalized 是否数据归一化
    @staticmethod
    def init_data(folder_path, metrics_num=20, is_sample=False, is_normalized=True,files_list = ['PC4.arff']):
        # 获取目录下所有原始文件
        if(files_list is None):
            files = os.listdir(folder_path)
        else:
            files = files_list
        # print("files",files)
        data_list, label_list = [], []
        for file in files:
            # 每一个子文件的真实路径
            print(file)
            file_path = folder_path+file
            type_file = os.path.splitext(file)[1]
            # txt文件
            if '.txt' == type_file  or '.TXT' == type_file :
                # 直接读取文件
                data_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=range(0, metrics_num+1))
                label_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=metrics_num+1)
                if is_normalized:
                    # 数据归一化
                    data_file -= data_file.min()
                    data_file /= data_file.max()
                    label_file -= label_file.min()
                    label_file /= label_file.max()
                # 加入列表
                data_list.append(data_file)
                label_list.append(label_file)
            # arff文件
            if '.arff' == type_file  or '.ARFF' == type_file :
                relation, attribute, data = [], [], []  # 保存arff 中的信息
                class_identify, t_class_identify, f_class_identify = [], '', ''
                is_first = True
                with open(file_path, 'r') as arff_file:
                    lines = arff_file.readlines()
                    for line in lines:
                        if '@relation' in line:
                            r = line.split(' ')
                            relation.append(tuple([r[0].replace('@', '').strip(), r[1].strip()]))
                            continue
                        if '@attribute' in line:
                            attribute.append(line.replace('@attribute', '').strip())
                            continue
                        if '\n' == line or '@data' in line:
                            continue
                        else:
                            if is_first:
                                # 如果第一次进来，通过判断已经读取的属性，将标签转化问1、0
                                class_identify = attribute[-1].split(' ')[1].replace('{', '').replace('}', '')
                                t_class_identify = class_identify.split(',')[0]
                                f_class_identify = class_identify.split(',')[1]
                                is_first = False
                            line_ = line.split(',')
                            class_TF = line_.pop(-1).replace('\n', '').strip()
                            if class_TF == t_class_identify:
                                class_TF = 1
                            if class_TF == f_class_identify:
                                class_TF = 0
                            line_.append(class_TF)
                            line_ = [float(i) for i in line_]
                            data.append(line_)
                            continue
                data = np.array(data)
                data = DataSetTool.clean(data)
                data_file = data[:, 0:-1]
                label_file = data[:, -1]
                if is_normalized:
                    # 数据归一化
                    data_file -= np.array(data_file).min()
                    data_file /= np.array(data_file).max()
                    label_file -= np.array(label_file).min()
                    label_file /= np.array(label_file).max()
                # 加入列表
                data_list.append(data_file)
                label_list.append(label_file)
        # 重采样
        if is_sample:
            for index in range(len(data_list)):
                data_list[index], label_list[index] = MAHAKIL().fit_sample(data_list[index], label_list[index])
        return data_list, label_list

    @staticmethod
    def get_positive_rate(data_list, label_list):
        for index in range(len(data_list)):
            positive = 0
            # 按照正例和反例划分数据集
            N = label_list[index].shape[0]  # 样例总数
            for i in range(N):
                if label_list[index][i] == 1:
                    positive += 1
            print(str(index) + ":positive rate is " + str(positive/N))

    #source源数据集，np.array格式
    #is_duplicate是否清除重复样本
    #is_zero_var是否清除0方差特征，即所有值都一样的特征
    #is_corr 是否清除相关系数大于阈值的特征
    #thr_corr相关系数阈值
    @staticmethod
    def clean(source,is_duplicate = True,is_zero_var = True,is_corr = True,thr_corr = 0.9):
        if (is_duplicate):
            print("去重前：",source.shape)
            source = np.array(list(set([tuple(t) for t in source])))
            print("去重后：",source.shape)

        feature = source[:,:-1]
        label = source[:,-1]
        if (is_zero_var):
            print("删除0方差特征前：",feature.shape)
            var = VarianceThreshold(threshold=0.0)   # 将方差小于等于1.0的特征删除。 默认threshold=0.0
            feature = var.fit_transform(feature)
            print("删除0方差特征后：",feature.shape)
        if (is_corr):
            list_corr = []
            for i in range(0,feature.shape[1]):
                for j in range(i+1,feature.shape[1]):
                    if (i not in list_corr and j not in list_corr):
                        a = feature[:,i]
                        b = feature[:,j]
                        ab = np.array([a, b])
                        if(np.corrcoef(ab)[0,1]>=thr_corr):
                            list_corr.append(j)
                        #返回的是相关系数矩阵，对角线是自身的相关系数1，用的是非对角线的
            print("需要删除的相关列序号:",list_corr)
            print("删除相关特征前：",feature.shape)
            feature = np.delete(feature, list_corr, axis=1)
            print("删除相关特征后：",feature.shape)
            label = label.reshape((label.shape[0],1))
            source = np.concatenate((feature,label),axis = 1)#列合并
            return source
            





