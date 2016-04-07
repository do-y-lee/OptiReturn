import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

## logistic regression
model_LR = LogisticRegression(penalty='l2')
model_LR.fit(X_train, y_train)
y_pred_LR = model_LR.predict(X_test)
LR_scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print metrics.accuracy_score(y_test, y_pred_LR)
print LR_scores
print LR_scores.mean()
print metrics.confusion_matrix(y_test, y_pred_LR)
print metrics.classification_report(y_test, y_pred_LR)
print model_LR.predict_proba(X_test[:10]) # 0, 1
print y_pred_LR[:10]

## coefficient analysis
X_colnames = []
for column in X_train.columns:
    X_colnames.append(column)

cnt = 0
coef_dict = {}
for coef in model_LR.coef_.tolist()[0]:
    if coef not in coef_dict.keys():
        coef_dict[X_colnames[cnt]] = coef
    cnt += 1

from collections import OrderedDict
feature_coef = OrderedDict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True))
feature_coef

## coefficient groupings
state = {}
grade = {}
subgrade = {}
home_ownership = {}
purpose = {}
emp_length = {}
metrics = {}

for k, v in feature_coef.iteritems():
    if 'state_' in k:
        state[k] = v
    elif 'subgrade_' in k:
        subgrade[k] = v
    elif 'home_ownership_' in k:
        home_ownership[k] = v
    elif 'purpose_' in k:
        purpose[k] = v
    elif 'grade_' in k:
        grade[k] = v
    elif 'emp_length_' in k:
        emp_length[k] = v
    elif v <= float(1):
        metrics[k] = v

