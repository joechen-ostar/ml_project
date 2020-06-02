import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('/cos_person/kobe_predict/kobe.csv')
data_cl = pd.read_csv('/cos_person/kobe_predict/kobe_transformation.csv')
target = data['shot_made_flag']

unknown_mask = data['shot_made_flag'].isnull()
X = data_cl[~unknown_mask]
y = target[~unknown_mask]

# Find all features with more than 90% variance in values.
threshold = 0.90
vt = VarianceThreshold().fit(X)
feat_var_threshold = data_cl.columns[vt.variances_ > threshold * (1-threshold)]


# Top 20 most important features According to RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feat_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index

# Univariate feature selection, Select top 20 features using  chi2  test. Features must be positive before applying test.
X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k='all').fit(X_minmax, y)
feature_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': X_scored.scores_
    })

feat_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values

# Recursive Feature Elimination, Select 20 features from using recursive feature elimination (RFE) with logistic regression model.
rfe = RFE(LogisticRegression(), 20)
rfe.fit(X, y)

feature_rfe_scoring = pd.DataFrame({
        'feature': X.columns,
        'score': rfe.ranking_
    })

feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values


# Finally features selected by all methods will be merged together
features = np.hstack([
        feat_var_threshold,
        feat_imp_20,
        feat_scored_20,
        feat_rfe_20
    ])

features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("\t-{}".format(f))

X = X.ix[:, features]
X.to_csv('/cos_person/kobe_predict/kobe_selection.csv', index=False, header=False)