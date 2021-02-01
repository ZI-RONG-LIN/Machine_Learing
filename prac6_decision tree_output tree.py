# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:17:44 2021

@author: T30518
"""


#decision tree_output tree

from sklearn.datasets import load_iris 
from sklearn import tree 
import graphviz 
X, y = load_iris(return_X_y=True) 
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 
tree.plot_tree(clf) 
dot_data= tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
#print(graph)


#精緻結果繪圖
dot_data= tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("iris") 