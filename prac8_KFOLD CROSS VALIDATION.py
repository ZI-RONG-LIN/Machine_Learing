# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:39:41 2021

@author: T30518
"""

#KFOLD CROSS VALIDATION K疊交互驗證
from sklearn.model_selection import KFold
from sklearn import tree
clf= tree.DecisionTreeClassifier() 
kf= KFold(n_splits=2) 
kf.get_n_splits(X) 
print(kf) 

for train_index, test_index in kf.split(X): 
    print("TRAIN:", train_index) 
    print("TEST:", test_index) 
    X_train, X_test= X[train_index], X[test_index] 
    y_train, y_test= y[train_index], y[test_index] 
    print("TRAIN data:") 
    print(X_train, y_train) 
    print("TEST data:") 
    print(X_test, y_test)
    clf= clf.fit(X_train, y_train) 
    predicted = clf.predict(X_test)
    tree.plot_tree(clf)
