# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:55:23 2021

@author: T30518
"""


#PREDICT NEW CASE_1
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
#from sklearn.tree import export_text
from sklearn.model_selection import cross_val_predict

iris = load_iris() 
X = iris.data 
y = iris.target
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy') 

print(scores) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
y_pred= cross_val_predict(clf, X, y, cv=10) 
for i in range(len(X)): 
    print(X[i], y_pred[i])
