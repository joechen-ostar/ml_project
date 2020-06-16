
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_curve, recall_score, precision_score
from sklearn.model_selection import train_test_split
from xunfei.product.ip2Region import Ip2Region
from sklearn.externals import joblib

# outline_path = "../data/round1_iflyad_anticheat_traindata.txt"
# online_path = "../data/round1_iflyad_anticheat_testdata_feature.txt"
outline_path = "../tmpdata/train.csv"
online_path = "../tmpdata/test.csv"
from datetime import timedelta, datetime

#todo 特征工程预留
def precession(df):
    df.drop(['imeimd5', 'openudidmd5', 'adidmd5', 'idfamd5','os','orientation','ppi'], 1, inplace=True)

    #TODO 分析一下缺失值，决定处理方式

    #TODO 特征ngixntime 时间戳 单独处理

    #对于连续数据做特征交叉
    df['hw'] = df['h'] * df['w']
    # df['hp'] = df['h'] * df['ppi']
    # df['pw'] = df['ppi'] * df['w']
    df['h_w'] = (df['h'] / df['w']).fillna(0.0)
    # df['hwp'] = df['h'] * df['w'] * df['ppi']

    # IP处理
    # def ip_func(se):
    #     tmp = se.split(".")[0]
    #     if len(tmp) > 3:
    #         num = 300
    #     else:
    #         num = int(tmp)
    #
    #     if num <= 126:  # 政府机构地址
    #         re = 0
    #     elif num <= 191:  # 大中型企业
    #         re = 1
    #     elif num <= 223:  # 个人
    #         re = 2
    #     elif num <= 239:  # 组播
    #         re = 3
    #     elif num <= 255:  # 研究
    #         re = 4
    #     else:
    #         re = 5
    #     return re

    # df_ci = df[['city', 'province','ip']]
    # df_citypro_ip = df['ip'].apply(ip2city)
    df.drop(['province'],1,inplace=True)
    # df = pd.concat([df,df_citypro_ip],axis=1)



    # df['reqrealip_deal'] = df['reqrealip'].apply(ip_func)
    # df['ip_deal'] = df['ip'].apply(ip_func)

    # 对model---设备进行处理
    df['model'].replace('PACM00', "OPPO R15", inplace=True)
    df['model'].replace('PBAM00', "OPPO A5", inplace=True)
    df['model'].replace('PBEM00', "OPPO R17", inplace=True)
    df['model'].replace('PADM00', "OPPO A3", inplace=True)
    df['model'].replace('PBBM00', "OPPO A7", inplace=True)
    df['model'].replace('PAAM00', "OPPO R15_1", inplace=True)
    df['model'].replace('PACT00', "OPPO R15_2", inplace=True)
    df['model'].replace('PABT00', "OPPO A5_1", inplace=True)
    df['model'].replace('PBCM10', "OPPO R15x", inplace=True)

    # df["model_count"] = df.groupby(['model'])['model'].transform('count')


    # 处理属性中出现的大小写问题
    df['model'] = df['model'].astype('str')
    df['model'] = df['model'].map(lambda x: x.upper())
    # df['os'] = df['os'].astype('str')
    # df['os'] = df['os'].map(lambda x: x.upper())

    # 在训练集中增加 day hour mintue
    df['datetime'] = pd.to_datetime(df['nginxtime'] / 1000, unit='s') + timedelta(
        hours=8)
    # pd.to_datatime
    df['hour'] = df['datetime'].dt.hour
    # df['day'] = df['datetime'].dt.day - df['datetime'].dt.day.min()
    df['minute'] = df['datetime'].dt.minute.astype('uint8')
    df.drop(['nginxtime'], axis=1, inplace=True)
    
    print(df.shape)
    # df.to_csv("../tmpdata/train.csv")
    return df

#数据读取  x1:a b c d a c  x1:1 2 3 4 1 3   x_on_1: a b c e
def load_data(path,path_on):
    df = pd.read_csv(path)
    df = precession(df)
    #分析了模型的feature_importance之后，删去了一些重要性很低的特征
    #todo （待优化）缺失值填充成了empty, 它是原本数据的缺失值的填充字符，与原本数据保持一致，填充之后也作为特征的一种属性
    df = df.fillna("empty")

    #线下训练数据
    x = df.drop(['label','sid'],axis=1)
    y = df['label']
    cols = x.columns

    #线上训练数据
    df_on = pd.read_csv(path_on)
    df_on = precession(df_on)
    x_on = df_on.drop(['sid'], axis=1)
    x_on = x_on.fillna("empty")
    #线上线下数据融合，防止编码过程中出现线下数据有编码，线上数据无编码的情况
    x_all = pd.concat([x, x_on], 0)
    print(x_all.shape)

    #把所有的字符编码成数字
    oe = OrdinalEncoder()
    oe.fit(x_all)
    x = oe.transform(x)
    print(x.shape)
    return x,y,oe,cols

#lightGbm训练
def trainModel(x,y):

    #lightgbm/xgboost的自定义评价指标
    def self_metric(y_true, y_pred):
        score = -f1_score(y_true, 1*(y_pred>=0.5))
        return 'f1', score, False

    from sklearn.ensemble import BaggingClassifier
    params = {"num_leaves":81, "n_estimators":550, "learning_rate":0.2,#绝对需要的参数
              # "subsample":0.9,
              "class_weight":{1:1,0:1},"reg_lambda":2 #仅做尝试
              }
    # params = {"num_leaves":121, "n_estimators":450, "learning_rate":0.2,#绝对需要的参数
    #           "subsample":0.9,"class_weight":{1:1,0:1},"reg_lambda":1 #仅做尝试
    #           }
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=345)
    lg = LGBMClassifier(**params)
    # lg = LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary',
    #                  learning_rate=0.1, n_estimators=2666, num_leaves=31, max_depth=-1,
    #                  min_child_samples=50, min_child_weight=9, subsample_freq=1,
    #                  # subsample=0.7, colsample_bytree=0.7,
    #                     reg_alpha=1, reg_lambda=5)
    model = BaggingClassifier(base_estimator=lg, n_estimators=100, max_samples=0.8, max_features=0.8)
    # model  = lg
    # model.fit(x_train, y_train, eval_metric=self_metric, eval_set=[(x_train, y_train),(x_test, y_test)],early_stopping_rounds=100)
    # model.n_estimators = model.best_iteration_
    model.fit(x_train, y_train)
    joblib.dump(model, "../result/lgb.m")

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
def blending(x,y,all_clumns):
    from xunfei.product.CatLgbBlending import CatLgbBlending
    num_clumns = ['h','w','ppi','hw','hp','pw','h_w','hwp']
    cat_clumns = set(all_clumns) - set(num_clumns)
    cates_idx = [all_clumns.values.tolist().index(c) for c in cat_clumns]
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=345)

    cb = CatLgbBlending()
    cb.fit(x,y,cates_idx)
    pre_train = cb.predict(x_train)
    pre_test = cb.predict(x_test)
    print(cb.thres)
    print("train f1_score", f1_score(y_train, pre_train))
    print("test f1_score", f1_score(y_test, pre_test))
    print("train recall_score", recall_score(y_train, pre_train))
    print("test recall_score", recall_score(y_test, pre_test))
    print("train precision_score", precision_score(y_train, pre_train))
    print("test precision_score", precision_score(y_test, pre_test))

    return cb
def predict(path, oe, model, the=0.5):


    # model = joblib.load("../result/lgb.m")
    df = pd.read_csv(path)
    df = precession(df)

    id = df['sid']
    x = df.drop(['sid'],1)
    x = x.fillna("empty")
    x = oe.transform(x)
    pred = model.predict_proba(x)[:,1]
    pred = 1*(pred>the)
    # pred = model.predict(x)
    pd.DataFrame({'sid':id,'label':pred}).to_csv("../result/resultAsLGB0816.csv", index=None)


def main():
    x, y, oh,columns = load_data(outline_path,online_path)
    model,the = trainModel(x, y)
    # model = blending(x, y,columns)
    predict(online_path, oh, model)

    # joblib.dump(model,"../result/cb.m")
    fi = model.feature_importances_
    for i, v in enumerate(fi):
        print(columns[i], v)
    # df = pd.read_csv(outline_path,sep="\t")
    # print(df['nginxtime'])
if __name__ == '__main__':
    main()