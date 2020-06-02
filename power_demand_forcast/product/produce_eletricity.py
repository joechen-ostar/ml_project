#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-19 13:59
# @Author  : Joe
# @Site    : 
# @File    : produce_eletricity.py
# @Software: PyCharm

# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/11
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

mpl.rcParams['font.sans-serif'] = [u'simHei']

# np.random.seed(0)

# 1. 加载数据(数据一般存在于磁盘或者数据库)
path = './feature_data_without_hour.csv'



df = pd.read_csv(path)

# 2. 数据清洗

# # 3. 根据需求获取最原始的特征属性矩阵X和目标属性Y
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
print(X.head())
print(X.head())
# print(Y)
# label_encoder = LabelEncoder()
# label_encoder.fit(Y)
# Y = label_encoder.transform(Y)
# 这里得到的序号其实就是classes_这个集合中对应数据的下标
# print(label_encoder.classes_)
# true_label = label_encoder.inverse_transform([0, 1, 2, 0])
# print(true_label)
# print(Y)

# 4. 数据分割
# train_size: 给定划分之后的训练数据的占比是多少，默认0.75
# random_state：给定在数据划分过程中，使用到的随机数种子，默认为None，使用当前的时间戳；给定非None的值，可以保证多次运行的结果是一致的。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练数据X的格式:{}, 以及类型:{}".format(x_train.shape, type(x_train)))
print("测试数据X的格式:{}".format(x_test.shape))
print("训练数据Y的类型:{}".format(type(y_train)))

# 5. 特征工程的操作
# NOTE: 不做特征工程

# 6. 模型对象的构建

algo = XGBRegressor(n_estimators=500,learning_rate=0.1, max_depth=8)

# 7. 模型的训练
algo.fit(x_train, y_train)

# 8. 模型效果评估
train_predict = algo.predict(x_train)
test_predict = algo.predict(x_test)

print("训练集上的mse:{}".format(mean_squared_error(y_train,train_predict)))
print("测试集上的mse:{}".format(mean_squared_error(y_test,test_predict)))

print("测试集上的效果(r_2):{}".format(algo.score(x_test, y_test)))
print("训练集上的效果(r_2):{}".format(algo.score(x_train, y_train)))
# print("测试集上的效果(分类评估报告):\n{}".format(classification_report(y_test, test_predict)))
# print("训练集上的效果(分类评估报告):\n{}".format(classification_report(y_train, train_predict)))

# 9. 其它
# print("返回的预测概率值:\n{}".format(algo.predict_proba(x_test)))
#
# # 10. 其他特殊的API
# print("各个特征属性的重要性权重:\n{}".format(algo.feature_importances_))

# # 返回叶子节点下标
# print("*" * 100)
# x_test2 = x_test.iloc[:2, :]
# print(x_test2)
# # apply方法返回的是叶子节点下标
# print(algo.apply(x_test2))
