# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:54:18 2021

@author: T30518
"""

#SVM-support vector machine 支援向量機器演算法
from sklearn.datasets import load_iris 
from sklearn import svm
from sklearn.model_selection import cross_val_score
#from sklearn.tree import export_text
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
iris = load_iris() 
X = iris.data 
y = iris.target 
clf= svm.SVC(kernel='linear', gamma=10) 
clf.fit(X, y) 
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(score) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
y_pred= cross_val_predict(clf, X, y, cv=10) 
conf_mat= confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred)) 
print(classification_report(y, y_pred))