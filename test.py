# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:19:43 2024

@author: usouu
"""

import utils
import covmap_construct as cc

# get label
labels = utils.get_label()

# get cm data
cmplv = utils.get_cmdata('PLV', 'sub1ex1')
cmpcc = utils.get_cmdata('PCC', 'sub1ex1')

cmplv = cmplv['cmplv_gamma']
cmpcc = cmpcc['cmpcc_gamma']

# # get sfcc
# sfcc_plv2 = cc.get_sfcc(cmplv)
# sfcc_pcc2 = cc.get_sfcc(cmpcc)

cvplv = cmplv.reshape(-1, cmplv.shape[1]**2)
cvpcc = cmplv.reshape(-1, cmpcc.shape[1]**2)

# svm
##########
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 假设 data 和 labels 已经定义
# data: shape (samples, features)
# labels: shape (samples,)

data = cvplv
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
