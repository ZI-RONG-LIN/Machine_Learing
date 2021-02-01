# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:43:07 2021

@author: T30518
"""


#CROSS VALIDATION

from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
iris = load_iris() 
X = iris.data 
y = iris.target 
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(score) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
