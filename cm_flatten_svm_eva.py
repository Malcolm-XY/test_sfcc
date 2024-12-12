# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:09:50 2024

@author: usouu
"""

import numpy

import utils
import covmap_construct as cc

# get label
labels = utils.get_label()

# get cm data
cmdata_pcc = utils.get_cmdata('PCC', 'sub1ex1')
cmdata_pcc_alpha = cmdata_pcc['cmpcc_alpha']
cmdata_pcc_beta = cmdata_pcc['cmpcc_beta']
cmdata_pcc_gamma = cmdata_pcc['cmpcc_gamma']
cmdata_pcc_alpha = cmdata_pcc_alpha.reshape(-1, cmdata_pcc_alpha.shape[1]**2)
cmdata_pcc_beta = cmdata_pcc_beta.reshape(-1, cmdata_pcc_beta.shape[1]**2)
cmdata_pcc_gamma = cmdata_pcc_gamma.reshape(-1, cmdata_pcc_gamma.shape[1]**2)
cmdata_pcc_joint = numpy.hstack((cmdata_pcc_alpha, cmdata_pcc_beta, cmdata_pcc_gamma))

# cmdata_plv = utils.get_cmdata('PLV', 'sub1ex1')
# cmdata_plv_alpha = cmdata_plv['cmplv_alpha']
# cmdata_plv_beta = cmdata_plv['cmplv_beta']
# cmdata_plv_gamma = cmdata_plv['cmplv_gamma']
# cmdata_plv_alpha = cmdata_plv_alpha.reshape(-1, cmdata_plv_alpha.shape[1]**2)
# cmdata_plv_beta = cmdata_plv_beta.reshape(-1, cmdata_plv_beta.shape[1]**2)
# cmdata_plv_gamma = cmdata_plv_gamma.reshape(-1, cmdata_plv_gamma.shape[1]**2)
# cmdata_pcc_joint = numpy.hstack((cmdata_plv_alpha, cmdata_plv_beta, cmdata_plv_gamma))

# svm
##########
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 假设 data 和 labels 已经定义
# data: shape (samples, features)
# labels: shape (samples,)

data = cmdata_pcc_joint
labels = utils.get_label()

# 顺序划分数据
split_index = int(0.7 * len(data))
data_train, data_test = data[:split_index], data[split_index:]
labels_train, labels_test = labels[:split_index], labels[split_index:]

# 创建并训练SVM分类器
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
svm_classifier.fit(data_train, labels_train)

# 测试模型
labels_pred = svm_classifier.predict(data_test)

# 输出分类报告和准确率
print("Classification Report:")
print(classification_report(labels_test, labels_pred))

accuracy = accuracy_score(labels_test, labels_pred)
print(f"Accuracy: {accuracy:.2f}")
