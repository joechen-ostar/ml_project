import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier

data = pd.read_csv('/cos_person/kobe_predict/kobe.csv')
target = data['shot_made_flag'].copy()
unknown_mask = data['shot_made_flag'].isnull()
y = target[~unknown_mask].iloc[1:]

X = pd.read_csv('/cos_person/kobe_predict/kobe_selection.csv')
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42)

seed = 7
estimators = []

estimators.append(('lr', LogisticRegression(penalty='l2', C=1)))
estimators.append(('gbm', GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, max_features=10, warm_start=True, random_state=seed)))
estimators.append(('rf', RandomForestClassifier(bootstrap=True, max_depth=8, n_estimators=200, max_features=20, criterion='gini', random_state=seed)))
estimators.append(('ada', AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1e-3, n_estimators=10, random_state=seed)))

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft', weights=[2,3,3,1])

model = ensemble

model.fit(X_train, y_train)
y_train_preds = model.predict(X_train)
y_preds = model.predict(X_test)

score = accuracy_score(y_train, y_train_preds)
print("train accuracy: " + str(score))

score = accuracy_score(y_test, y_preds)
print("test accuracy: " + str(score))

result = pd.DataFrame()
result["true"] = y_test
result["pred"] = y_preds

result.to_csv("/cos_person/kobe_predict/result.csv", index=False, header=False, sep=" ")