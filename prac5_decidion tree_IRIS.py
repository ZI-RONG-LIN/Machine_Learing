# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:16:20 2021

@author: T30518
"""


#decidion tree_IRIS

from sklearn.datasets import load_iris 
from sklearn import tree 
#import graphviz 
#載入IRIS的資料  TRUE的話會回傳xy回來
X, y = load_iris(return_X_y=True) 
'''
上面那句等同於
iris = load_iris()
X=iris.data
y=iris.target
'''
#呼叫決策樹模型
clf= tree.DecisionTreeClassifier() 
#把xy放入模型中
clf= clf.fit(X, y) 
#印出樹，屬於tree的繪圖函數，其他演算法要另外處理
tree.plot_tree(clf)