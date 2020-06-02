
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder # 针对类别特征处理
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_curve, recall_score, precision_score
from sklearn.model_selection import train_test_split

outline_path = "../data/round1_iflyad_anticheat_traindata.txt"
online_path = "../data/round1_iflyad_anticheat_testdata_feature.txt"

#todo 特征工程预留
def precession(df):
    #TODO 分析一下缺失值，决定处理方式

    #TODO 特征ngixntime 时间戳 单独处理

    #对于连续数据做特征交叉
    df['hw'] = df['h'] * df['w']
    df['hp'] = df['h'] * df['ppi']
    df['pw'] = df['ppi'] * df['w']
    df['h_w'] = (df['h'] / df['w']).fillna(0.0) # 这里填充的类型为 np.nan
    df['hwp'] = df['h'] * df['w'] * df['ppi']

    # IP处理
    def ip_func(se):
        tmp = se.split(".")[0]
        if len(tmp) > 3:
            num = 300
        else:
            num = int(tmp)

        if num <= 126:  # 政府机构地址
            re = 0
        elif num <= 191:  # 大中型企业
            re = 1
        elif num <= 223:  # 个人
            re = 2
        elif num <= 239:  # 组播
            re = 3
        elif num <= 255:  # 研究
            re = 4
        else:
            re = 5
        return re

    df['reqrealip_deal'] = df['reqrealip'].apply(ip_func)
    df['ip_deal'] = df['ip'].apply(ip_func)

    return df

#数据读取  x1:a b c d a c  x1:1 2 3 4 1 3   x_on_1: a b c e
def load_data(path,path_on):
    df = pd.read_csv(path, sep="\t")
    df = precession(df)
    #分析了模型的feature_importance之后，删去了一些重要性很低的特征
    df.drop(['imeimd5', 'openudidmd5', 'os', 'adidmd5', 'idfamd5'], 1, inplace=True)
    #todo （待优化）缺失值填充成了empty, 它是原本数据的缺失值的填充字符，与原本数据保持一致，填充之后也作为特征的一种属性
    df = df.fillna("empty")

    #线下训练数据
    x = df.drop(['label','sid'],axis=1)
    y = df['label']
    cols = x.columns

    #线上训练数据
    df_on = pd.read_csv(path_on, sep="\t")
    df_on = precession(df_on)
    df_on.drop(['imeimd5', 'openudidmd5', 'os', 'adidmd5', 'idfamd5'], 1, inplace=True)
    x_on = df_on.drop(['sid'], axis=1)
    x_on = x_on.fillna("empty")
    #线上线下数据融合，防止编码过程中出现线下数据有编码，线上数据无编码的情况
    x_all = pd.concat([x, x_on], 0)

    #把所有的字符编码成数字
    oe = OrdinalEncoder()
    oe.fit(x_all)  # 直接传入 他会自动将object类型换掉
    x = oe.transform(x)
    print(x.shape)
    return x,y,oe,cols

#lightGbm训练
def trainModel(x,y):

    #lightgbm/xgboost的自定义评价指标
    def self_metric(y_true, y_pred):
        score = f1_score(y_true, 1*(y_pred>=0.5))  # 因为传入进去的y_true和y_pred必须是二进制的数据类型，因此需要
        return 'f1', score, False

    from sklearn.ensemble import BaggingClassifier
    params = {"num_leaves":81, "n_estimators":100, "learning_rate":0.2,#绝对需要的参数
              "subsample":0.9,"class_weight":{1:1,0:1},"reg_lambda":2 #仅做尝试
              }
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=345)
    lgb = LGBMClassifier(**params)
    """
    boosting_type='gbdt', 默认是 gdbt 梯度提升树 dart dropout和MArt的结合 后者就是多层加法树模型(multiple additive regression tree）goss（基于梯度的单侧采样） rf 随机森林
    num_leaves=31, 基础学习器最多的叶子节点
    max_depth=-1,  基础学习器的最大树深  小于等于0意味着没有限制
    learning_rate=0.1, boosting的缩放系数
    n_estimators=100, 学习器的数量
    subsample_for_bin=200000, 多少样本构建分箱
    objective=None, 指定具体的任务类型 如果是是分类就是 binary或者multiple 回归就是regression 排序就是lambdarank
    class_weight=None, 样本权重 不同类别的样本权重可能不一样
    min_split_gain=0., 
    min_child_weight=1e-3, 
    min_child_samples=20,
    subsample=1., 
    subsample_freq=0, 
    colsample_bytree=1.,
    reg_alpha=0., 
    reg_lambda=0.,
    random_state=None,
    n_jobs=-1, 
    silent=True, 
    importance_type='split'
    
    
    """
    # model = BaggingClassifier(base_estimator=lg, n_estimators=100, max_samples=0.8, max_features=0.8)
    model  = lgb
    model.fit(x_train, y_train, eval_metric=self_metric, eval_set=[(x_train, y_train),(x_test, y_test)]) # 默认没有F1指标 所以自定义
    # model.fit(x_train, y_train)
    """
    sample_weight=None, 
    init_score=None,
    eval_set=None, 
    eval_names=None, 
    eval_sample_weight=None,
    eval_class_weight=None, 
    eval_init_score=None, 
    eval_metric=None,
    early_stopping_rounds=None,  正常的是应该在测试集效果越来越好，如果连续n轮效果越来越差 就提前结束训练
    verbose=True,
    feature_name='auto', 
    categorical_feature='auto', 
    callbacks=None
    
    
    
    """

    """
    如果设置early_stopping这个参数 那么要将迭代的轮数回传给模型，把迭代效果不好的轮次不要掉 model.n_estimators = model.best_iteration_
    
    """

    #质变部分 - 取合理的阈值来指定 f1指标
    #todo 可以自己划分多个阈值（2000个以上）直接计算f1指标，看哪个阈值最好，更加精确
    pre_train = model.predict_proba(x_train)[:,1]
    pre_test = model.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_train, pre_train)
    thre_index = (tpr - fpr).argmax()
    thres = thresholds[thre_index]


    print("训练集阈值",thres)
    pre_train = 1*(pre_train>=thres)
    pre_test = 1 * (pre_test >= thres)
    print("train f1_score",f1_score(y_train, pre_train))
    print("test f1_score", f1_score(y_test, pre_test))
    print("train recall_score",recall_score(y_train, pre_train))
    print("test recall_score", recall_score(y_test, pre_test))
    print("train precision_score",precision_score(y_train, pre_train))
    print("test precision_score", precision_score(y_test, pre_test))
    return model,thres

def predict(path, oe, model, the):
    df = pd.read_csv(path, sep="\t")
    df = precession(df)
    df.drop(['imeimd5', 'openudidmd5', 'os', 'adidmd5', 'idfamd5'], 1, inplace=True)

    id = df['sid']
    x = df.drop(['sid'],1)
    x = x.fillna("empty")
    x = oe.transform(x)
    pred = model.predict_proba(x)[:,1]
    pred = 1*(pred>the)
    pd.DataFrame({'sid':id,'label':pred}).to_csv("../result/resultAsLGB0722.csv", index=None)


def main():
    x, y, oh,columns = load_data(outline_path,online_path)
    model,the = trainModel(x, y)
    fi = model.feature_importances_
    for i, v in enumerate(fi):
        print(columns[i], v)
    predict(online_path, oh, model,the)
    # df = pd.read_csv(outline_path,sep="\t")
    # print(df['nginxtime'])
if __name__ == '__main__':
    # main()
    y_pred=[0.3,0.4,0.6,0.9,1.0]
    print(1*(y_pred>=0.5))
