# 平衡数据集
#X：数据，y：标签（pos：1，neg：0）

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import math
import joblib
from sklearn import datasets
from sklearn.model_selection import RepeatedStratifiedKFold,GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier,VotingClassifier,GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, matthews_corrcoef,make_scorer,recall_score,auc
from sklearn.metrics import roc_curve, roc_auc_score
from csv import reader
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from mlxtend.classifier import StackingClassifier

file = 'G:/features/SMOTE.csv'
# file = 'G:/features/no_CFS_SMOTE.csv'
# file = 'G:/features/polynom_fit_SMOTE.csv'
# file = 'G:/features/ProWSyn.csv'
# file = 'G:/features/SMOTE_IPF.csv'
# file = 'G:/features/kmeans_SMOTE.csv'

with open(file, 'rt', encoding='UTF-8-sig') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    X = np.array(x).astype('float')
y = np.zeros(1520)
for i in range(760):
    y[i] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


log_clf = LogisticRegression(penalty='l2',C=500,max_iter=1000,solver='newton-cg')    #,solver='newton-cg'
rnd_clf = RandomForestClassifier(n_estimators=150)   #"criterion":["gini","entropy"],,criterion='gini',min_samples_leaf=2
svm_clf = SVC(probability=True,C=550)   #"kernel":({'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}),gamma=0.1,kernel='rbf'
knn_clf = KNeighborsClassifier(n_neighbors=15)   #"weights":("uniform","distance")
bag_clf = BaggingClassifier(n_estimators=500)    #,max_samples=0.5
gbdt_clf = GradientBoostingClassifier()
lgb_clf = LGBMClassifier(learning_rate=0.033)
xgb_clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.06, max_delta_step=0,
              max_depth=6, min_child_weight=1,
              monotone_constraints='()', n_estimators=100, n_jobs=4,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=None)
ab_clf = AdaBoostClassifier(learning_rate=0.22)


# 集成以上几种分类器的投票分类器
voting_clf = VotingClassifier(
    estimators=[('xgb',xgb_clf),('ab',ab_clf),('lgb',lgb_clf),('gbdt',gbdt_clf),('rf',rnd_clf),('bag',bag_clf),('knn',knn_clf)],
    voting='soft'
)


rskf = RepeatedStratifiedKFold(n_splits=10,n_repeats=2)
accs = []
Sns = []
Sps = []
MCCs = []
precisions = []
recalls = []
Fs = []
Gs = []
aucs = []
tprs = []
fprs = []
mean_fpr = np.linspace(0,1,100)

for train_index, test_index in rskf.split(X, y):
    voting_clf.fit(X[train_index], y[train_index])
    y_score = voting_clf.predict_proba(X[test_index])[:, 1]
    pred_y_ = voting_clf.predict(X[test_index])

    #计算指标
    TP = np.sum(np.logical_and(np.equal(y[test_index], 1), np.equal(pred_y_, 1)))
    FP = np.sum(np.logical_and(np.equal(y[test_index], 0), np.equal(pred_y_, 1)))
    FN = np.sum(np.logical_and(np.equal(y[test_index], 1), np.equal(pred_y_, 0)))
    TN = np.sum(np.logical_and(np.equal(y[test_index], 0), np.equal(pred_y_, 0)))
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    MCC = matthews_corrcoef(y[test_index], pred_y_)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F = (2 * precision * recall) / (precision + recall)
    G = math.sqrt(Sn * Sp)
    acc = accuracy_score(y[test_index], pred_y_)
    accs.append(acc)
    Sns.append(Sn)
    Sps.append(Sp)
    MCCs.append(MCC)
    precisions.append(precision)
    recalls.append(recall)
    Fs.append(F)
    Gs.append(G)
    print("混淆矩阵：\n", confusion_matrix(y[test_index], pred_y_,labels=[1, 0]))

    # ROC
    fpr, tpr, thresholds = roc_curve(y[test_index], y_score)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    std = np.std(roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs, axis=0)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, label='%s (AUC:%0.4f ± %0.3f)' % ("ROC_AUC", mean_auc, std_auc), ls='-', lw=1, color='blue')
# plt.plot(C, mean_auc, ls='-', lw=2, color='darkgreen')

print('ACC:\t', np.mean(accs))
print('Sn:\t', np.mean(Sns))
print('Sp:\t', np.mean(Sps))
print('MCC:\t', np.mean(MCCs))
print('precision:\t', np.mean(precisions))
print('recall:\t', np.mean(recalls))
print('F:\t', np.mean(Fs))
print('G:\t', np.mean(Gs))

print('\n')

plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right', edgecolor='grey', fontsize=8)
plt.show()


# 保存模型
if mean_auc > 0.9 and np.mean(accs) > 0.9:
    name = "G:/model/修改voting_" + str(mean_auc) + ".m"
    joblib.dump(voting_clf, name)
    print("saved!")

# name = "G:/model/1_AB_" + str(mean_auc) + ".m"
# joblib.dump(log_clf, name)
# print("saved!")
