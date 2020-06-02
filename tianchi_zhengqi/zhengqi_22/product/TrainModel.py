
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

online_path = "../data/zhengqi_test.txt" #线上预测的数据，用户预测结果并上传
outline_path = "../data/zhengqi_train.txt" #线下训练的数据，用于训练模型
#读取数据和基础清洗
def load_data(path):
    return pd.read_csv(path, sep="\t")
#特征工程-选择不用交叉验证
def engineering(df, d):
    #训练集测试集划分
    x = df.drop(['target'],1)
    y = df['target']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=105)
    #多项式扩展
    poly = PolynomialFeatures(degree=d,interaction_only=False,include_bias=True)
    x_train = poly.fit_transform(x_train)
    x_test = poly.fit_transform(x_test)
    #数据标准化 todo以后可以加
    return x_train,x_test,y_train,y_test,poly
#模型训练
def train_model(x_train,x_test,y_train,y_test):

    # linear = ElasticNetCV(l1_ratio=0.5, alphas=[0.05,0.1,0.5,1,5])
    # linear.fit(x_train, y_train)

    # print("最好的alpha",linear.alpha_,"\n")
    """
    最好的alpha 0.05  l1_ratio=0.5   成绩  mse_train:0.127495  mse_test:0.131905
    """

    model = RandomForestRegressor(n_estimators=100,max_depth=10, max_features=0.2, random_state=100,min_samples_leaf=1,min_samples_split=2)
    model.fit(x_train,y_train)
    """
    d=2
    n_estimators=10,max_depth=1,random_state=100,min_samples_leaf=1,min_samples_split=2   成绩 mse_train:0.383068     mse_test:0.392434
    n_estimators=10,max_depth=3,random_state=100,min_samples_leaf=1,min_samples_split=2   成绩 mse_train:0.159486     mse_test:0.178690
    n_estimators=10,max_depth=5,random_state=100,min_samples_leaf=1,min_samples_split=2   成绩 mse_train:0.099685     mse_test:0.144653
    n_estimators=100,max_depth=5,random_state=100,min_samples_leaf=1,min_samples_split=2  成绩 mse_train:0.095347     mse_test:0.144288
    n_estimators=100,max_depth=8, max_features=0.2, random_state=100,min_samples_leaf=1,min_samples_split=2  成绩 mse_train:0.048788     mse_test:0.132891
    n_estimators=100,max_depth=10, max_features=0.2, random_state=100,min_samples_leaf=1,min_samples_split=2 成绩 mse_train:0.030698     mse_test:0.131680
    d=3
    n_estimators=100,max_depth=10, max_features=0.2, random_state=100,min_samples_leaf=1,min_samples_split=2 成绩 mse_train:0.025229     mse_test:0.129458
    
    """

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    print("成绩 mse_train:%f     mse_test:%f" %(mse_train, mse_test))
    return model

def prediction(online_path, poly, model):
    x = load_data(online_path)
    x = poly.transform(x)
    pred_online = model.predict(x)
    pd.DataFrame(pred_online).to_csv("../result/20200105.txt", header=None,index=None)

#主函数
def main():
    d = 3
    df = load_data(outline_path)
    # print(df.isnull().sum())  结果表明没有缺失值
    x_train,x_test,y_train,y_test,poly = engineering(df, d)
    # print(x_train)
    model = train_model(x_train,x_test,y_train,y_test)
    prediction(online_path, poly, model)

if __name__ == '__main__':
    main()
