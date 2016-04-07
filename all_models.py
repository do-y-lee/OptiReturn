import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

%matplotlib inline
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


## Logistic Regression + GridSearch
modelLR_grid_search = GridSearchCV(LogisticRegression(), param_grid={'C': np.logspace(-3, 3, 7)})
modelLR_grid_search.fit(X_train, y_train)
y_pred_LRGS = modelLR_grid_search.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_LRGS)

print metrics.confusion_matrix(y_test, y_pred_LRGS)
print metrics.classification_report(y_test, y_pred_LRGS)

modelLR_grid_search.best_params_
modelLR_grid_search.grid_scores_
modelLR_grid_search.best_score_


## Gaussian Naive Bayes
modelGNB = GaussianNB()
modelGNB.fit(X_train, y_train)
y_pred_GNB = modelGNB.predict(X_test)
gnb_cv_scores = cross_val_score(GaussianNB(), X, y, scoring='accuracy', cv=10)

print metrics.accuracy_score(y_test, y_pred_GNB)
print gnb_cv_scores
print gnb_cv_scores.mean()
print metrics.confusion_matrix(y_test, y_pred_GNB)
print metrics.classification_report(y_test, y_pred_GNB)


## Decision Tree
modelDTC = DecisionTreeClassifier()
modelDTC.fit(X_train, y_train)
y_pred_dtc = modelDTC.predict(X_test)
dtc_cv_scores = cross_val_score(DecisionTreeClassifier(), X, y, scoring='accuracy', cv=10)

print metrics.accuracy_score(y_test, y_pred_dtc)
print dtc_cv_scores
print dtc_cv_scores.mean()
print metrics.confusion_matrix(y_test, y_pred_dtc)
print metrics.classification_report(y_test, y_pred_dtc)


## Random Forest
modelRFC = RandomForestClassifier()
modelRFC.fit(X_train, y_train)
y_pred_rfc = modelRFC.predict(X_test)
rfc_cv_scores = cross_val_score(RandomForestClassifier(), X, y, scoring='accuracy', cv=10)

print metrics.accuracy_score(y_test, y_pred_rfc)
print rfc_cv_scores
print rfc_cv_scores.mean()
print metrics.confusion_matrix(y_test, y_pred_rfc)
print metrics.classification_report(y_test, y_pred_rfc)


## Bernoulli Naive Bayes
modelBNB = BernoulliNB()
modelBNB.fit(X_train, y_train)
y_pred_bnb = modelBNB.predict(X_test)
bnb_cv_scores = cross_val_score(BernoulliNB(), X, y, scoring='accuracy', cv=10)

print metrics.accuracy_score(y_test, y_pred_bnb)
print bnb_cv_scores
print bnb_cv_scores.mean()
print metrics.confusion_matrix(y_test, y_pred_bnb)
print metrics.classification_report(y_test, y_pred_bnb)



################ ROC curves #################

## logistic regression
fpr, tpr, thresholds = roc_curve(y_test, y_pred_LR)
roc_auc = auc(fpr, tpr)
label = 'LogisticRegression' + ' ROC Curve (AUC: %.3f)' % roc_auc
plt.subplots(1,1,figsize=(10,8))
plt.plot(fpr, tpr, label=label, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)

## random forest
fpr, tpr, thresholds = roc_curve(y_test, y_pred_rfc)
roc_auc = auc(fpr, tpr)
label = 'RandomForest' + ' ROC Curve (AUC: %.3f)' % roc_auc
plt.subplots(1,1,figsize=(10,8))
plt.plot(fpr, tpr, label=label, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)


## GaussianNB
fpr, tpr, thresholds = roc_curve(y_test, y_pred_GNB)
roc_auc = auc(fpr, tpr)
label = 'GaussianNB' + ' ROC Curve (AUC: %.3f)' % roc_auc
plt.subplots(1,1,figsize=(10,8))
plt.plot(fpr, tpr, label=label, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)


## Bernoulli Naive Bayes V1
fpr, tpr, thresholds = roc_curve(y_test, y_pred_bnb)
roc_auc = auc(fpr, tpr)
label = 'BernoulliNB' + ' ROC Curve (AUC: %.3f)' % roc_auc
plt.subplots(1,1,figsize=(10,8))
plt.plot(fpr, tpr, label=label, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)










