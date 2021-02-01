# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:07:07 2021

@author: T30518
"""
#SAVE AND LOAD MODEL
#儲存MODEL
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
import pickle

iris = load_iris() 
X = iris.data 
y = iris.target 
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(score) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
pkl_filename= "iris_model.pkl"
 #寫二進制文件 
with  open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
#讀二進制文件 
with open(pkl_filename,'rb') as file: 
    pickle_model= pickle.load(file)
newX= [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]] 
print(pickle_model.predict(newX))
