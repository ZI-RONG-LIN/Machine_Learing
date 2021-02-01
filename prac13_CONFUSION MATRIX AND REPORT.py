# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:05:13 2021

@author: T30518
"""



#CONFUSION MATRIX AND REPORT
#建立錯差矩陣，並輸出
from sklearn.datasets import load_iris 
from sklearn import tree 
from sklearn.model_selection import cross_val_score
#from sklearn.tree import export_text
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import pickle
iris = load_iris() 
X = iris.data 
y = iris.target
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy') 


print(scores) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
y_pred= cross_val_predict(clf, X, y, cv=10) 
conf_mat= confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred)) 
print(classification_report(y, y_pred))
#儲存MODEL
#wb的b適用2進制進行儲存
pkl_filename= "iris_model.pkl"
with  open(pkl_filename, 'wb') as file: #寫二進制文件 
    pickle.dump(clf, file)
