#X：数据，y：标签（pos：1，neg：0）

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, matthews_corrcoef,make_scorer,recall_score,auc
from csv import reader
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import math
import joblib


file = 'G:/features/independent_test.csv'
print(file)

with open(file, 'rt', encoding='UTF-8-sig') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    X = np.array(x).astype('float')
y = np.zeros(221)
for i in range(31):
    y[i] = 1

log_clf = joblib.load('G:/model/Logistic_0.9311980609418281.m')
rnd_clf = joblib.load('G:/model/Random_0.9909539473684209.m')
svm_clf = joblib.load('G:/model/SVM_0.9673303324099723.m')
knn_clf = joblib.load('G:/model/KNN_0.9296052631578947.m')
bag_clf = joblib.load('G:/model/BAG_0.9906682825484765.m')
gbdt_clf = joblib.load('G:/model/GBDT_0.9964162049861496.m')
lgb_clf = joblib.load('G:/model/LGB_0.9962950138504155.m')
xgb_clf = joblib.load('G:/model/XGB_0.9959660664819945.m')
ab_clf = joblib.load('G:/model/AdaB_0.9967191828254848.m')
voting_clf = joblib.load('G:/model/voting_XGALRBK_0.9967797783933519.m')


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
specificity = make_scorer(recall_score,pos_label=0,greater_is_better=True)
# accs = []
# Sns = []
# Sps = []
# MCCs = []
# precisions = []
# recalls = []
# Fs = []
# Gs = []
# # rskf = cv
aucs = []
tprs = []
fprs = []
mean_fpr = np.linspace(0,1,100)
colors = [
          'hotpink',
          'darkorange',
          'gold',
          'green',
          'steelblue',
          'purple',
          'skyblue',
          'cyan',
          'saddlebrown',
          # 'red'
          ]
clfs = [
        log_clf,
        rnd_clf,
        svm_clf,
        knn_clf,
        bag_clf,
        gbdt_clf,
        lgb_clf,
        xgb_clf,
        ab_clf,
        # voting_clf
        # sclf
        ]
names = [
        'Logistic',
        'RF',
        'SVM',
        'KNN',
        'BAG',
        'GBDT',
        'LightGBM',
        'XGBoost',
        'AdaBoost',
        # 'Voting'
        ]



# for clf in (log_clf, rnd_clf, svm_clf, knn_clf, bag_clf, voting_clf):
for (clf, colorname, name) in zip(clfs, colors, names):

    # for train_index, test_index in rskf.split(X1, y1):
    # clf.fit(X, y)
    y_score = clf.predict_proba(X)[:, 1]
    pred_y_ = clf.predict(X)

    #计算模型评估指标
    TP = np.sum(np.logical_and(np.equal(y, 1), np.equal(pred_y_, 1)))
    FP = np.sum(np.logical_and(np.equal(y, 0), np.equal(pred_y_, 1)))
    FN = np.sum(np.logical_and(np.equal(y, 1), np.equal(pred_y_, 0)))
    TN = np.sum(np.logical_and(np.equal(y, 0), np.equal(pred_y_, 0)))
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    MCC = matthews_corrcoef(y, pred_y_)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = (2 * precision * recall) / (precision + recall)
    G = math.sqrt(Sn * Sp)
    acc = accuracy_score(y,pred_y_)
    # accs.append(acc)
    # Sns.append(Sn)
    # Sps.append(Sp)
    # MCCs.append(MCC)
    # precisions.append(precision)
    # recalls.append(recall)
    # Fs.append(F)
    # Gs.append(G)

    #画ROC曲线
    fpr, tpr, thresholds = roc_curve(y, y_score)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    area = auc(fpr, tpr)
    aucs.append(area)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc1 = auc(mean_fpr, mean_tpr)
    mean_auc2 = np.mean(aucs, axis=0)
    # std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, label='%s (AUC:%0.4f)' % (name, mean_auc2), ls='--',
             lw=1, color=colorname)


    print(name)
    print('TP,TN,FN,FP: ',TP,TN,FN,FP)
    print('ACC:\t', acc)
    print('Sn:\t', Sn)
    print('Sp:\t', Sp)
    print('MCC:\t', MCC)
    print('precision:\t', precision)
    print('recall:\t', recall)
    print('F:\t', F)
    print('G:\t', G)
    print('AUC: ',area)
    print("混淆矩阵：", confusion_matrix(y, pred_y_,labels=[1,0]))
    print('\n')

#VOTING
# voting_clf.fit(X, y)
y_score = voting_clf.predict_proba(X)[:, 1]
pred_y_ = voting_clf.predict(X)

#计算模型评估指标
TP = np.sum(np.logical_and(np.equal(y, 1), np.equal(pred_y_, 1)))
FP = np.sum(np.logical_and(np.equal(y, 0), np.equal(pred_y_, 1)))
FN = np.sum(np.logical_and(np.equal(y, 1), np.equal(pred_y_, 0)))
TN = np.sum(np.logical_and(np.equal(y, 0), np.equal(pred_y_, 0)))
Sn = TP / (TP + FN)
Sp = TN / (TN + FP)
MCC = matthews_corrcoef(y, pred_y_)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F = (2 * precision * recall) / (precision + recall)
G = math.sqrt(Sn * Sp)
acc = accuracy_score(y,pred_y_)

#画ROC曲线
fpr, tpr, thresholds = roc_curve(y, y_score)
tprs.append(np.interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
area = auc(fpr, tpr)
aucs.append(area)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc1 = auc(mean_fpr, mean_tpr)
mean_auc2 = np.mean(aucs, axis=0)

plt.plot(mean_fpr, mean_tpr, label='ECAmyloid (AUC:%0.4f)' % (mean_auc2), ls='-',lw=1.5, color='red')
print("Voting")
print('TP,TN,FN,FP: ', TP, TN, FN, FP)
print('ACC:\t', acc)
print('Sn:\t', Sn)
print('Sp:\t', Sp)
print('MCC:\t', MCC)
print('precision:\t', precision)
print('recall:\t', recall)
print('F:\t', F)
print('G:\t', G)
print('AUC: ', area)
print("混淆矩阵：", confusion_matrix(y, pred_y_, labels=[1, 0]))

plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('independent_test ROC')
plt.legend(loc='lower right', edgecolor='grey', fontsize=10)
plt.show()
