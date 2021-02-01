# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:57:12 2021

@author: T30518
"""

#PREDICT NEW CASE_2
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict
from sklearn.tree import export_text
iris = load_iris() 
X = iris.data 
y = iris.target
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy') 
print(scores) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
newX= [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]] 
print(clf.predict(newX)) 
tree_rules= export_text(clf, feature_names=iris['feature_names'])