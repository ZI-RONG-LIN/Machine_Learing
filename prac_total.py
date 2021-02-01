# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:03:12 2021

@author: T30518
"""

#練習
#綜合上述功能，進行IRIS的decision tree分析 
#1.資料觀察 
#2.讀進資料，建立模型 
#3.進行cross validation 
#4. 列印錯差矩陣(confusion matrix) 
#5. 列印錯差矩陣的性能指標 
#6.列印決策樹 
#7.列印決策規則 
#8.預測新案例的分類結果： 
#newX= [[3.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]

#-----------------
#1.資料觀察 
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt
from sklearn import tree 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import export_text
import graphviz 
iris = load_iris() 
X = iris.data 
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()
# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

#2.讀進資料，建立模型
clf= tree.DecisionTreeClassifier() 
clf= clf.fit(X, y) 

#3.進行cross validation
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(score) 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

#4. 列印錯差矩陣(confusion matrix) 
#5. 列印錯差矩陣的性能指標 
y_pred= cross_val_predict(clf, X, y, cv=10) 
conf_mat= confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred)) 
print(classification_report(y, y_pred))

#色塊
dot_data= tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,filled=True, rounded=True,special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("iris") 


#6.列印決策樹 
tree.plot_tree(clf)

#7.列印決策規則 
tree_rules= export_text(clf, feature_names=iris['feature_names']) 
print(tree_rules)

#8.預測新案例的分類結果： 
newX= [[3.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
print(clf.predict(newX)) 