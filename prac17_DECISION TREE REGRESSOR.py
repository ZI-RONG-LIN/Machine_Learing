# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:44:30 2021

@author: T30518
"""

#迴歸樹分類(DECISION TREE REGRESSOR) 
from sklearn.datasets import load_iris 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

iris = load_iris() 
X = iris.data 
y = iris.target 
clf= DecisionTreeRegressor(random_state=0) 
clf= clf.fit(X, y) 
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy') 
print(score) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))