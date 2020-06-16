import catboost as cb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,roc_curve
import numpy as np
from sklearn.linear_model import LogisticRegression

class CatLgbBlending:
    # def __init__(self):
    #     return 0

    def fit(self,x,y,cate_index):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=345)

        cls = cb.CatBoostClassifier(
            iterations=400,
            od_type='Iter',
            od_wait=50,
            max_depth=5,
            learning_rate=0.1,
            l2_leaf_reg=9,
            random_seed=2019,
            # metric_period=10,
            eval_metric='F1',
            fold_len_multiplier=1.1,
            loss_function='Logloss',
            logging_level='Verbose')
        cls.fit(x_train, y_train, eval_set=(x_test, y_test), cat_features=cate_index)

        def self_metric(y_true, y_pred):
            score = -f1_score(y_true, 1 * (y_pred >= 0.5))
            return 'f1', score, False

        lg = LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary',
                         learning_rate=0.1, n_estimators=6000, num_leaves=31, max_depth=-1,
                         min_child_samples=50, min_child_weight=9, subsample_freq=1,
                         subsample=0.7, colsample_bytree=0.7,
                            reg_alpha=1, reg_lambda=5)
        lg.fit(x_train, y_train, eval_metric=self_metric, eval_set=[(x_train, y_train), (x_test, y_test)],
               early_stopping_rounds=200)
        lg.n_estimators = lg.best_iteration_

        train_prob1 = cls.predict_proba(x_train)[:,1].reshape(-1,1)
        train_prob2 = lg.predict_proba(x_train)[:,1].reshape(-1,1)
        train_prob = np.hstack([train_prob1,train_prob2])
        lr = LogisticRegression(C=10)
        lr.fit(train_prob, y_train)

        train_prob_lr = lr.predict_proba(train_prob)[:,1]
        fpr, tpr, thresholds = roc_curve(y_train, train_prob_lr)
        thre_index = (tpr - fpr).argmax()
        thres = thresholds[thre_index]
        self.thres = thres

        self.m1 = cls
        self.m2 = lg
        self.m3 = lr

    def predict(self,x):
        cls = self.m1
        lg = self.m2
        lr = self.m3
        thres = self.thres
        prob1 = cls.predict_proba(x)[:,1].reshape(-1,1)
        prob2 = lg.predict_proba(x)[:,1].reshape(-1,1)
        prob = np.hstack([prob1, prob2])
        prob_lr = lr.predict_proba(prob)[:,1]
        pre_train = 1 * (prob_lr >= thres)

        return pre_train


