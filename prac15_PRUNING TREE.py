# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:30:38 2021

@author: T30518
"""
#PRUNING TREE

from sklearn.datasets import load_iris 
from sklearn import tree 
import graphviz 
X, y = load_iris(return_X_y=True) 
clf= tree.DecisionTreeClassifier(min_samples_leaf=3) 
clf= clf.fit(X, y) 
tree.plot_tree(clf) 
dot_data= tree.export_graphviz(clf,  out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("irislimit3") 