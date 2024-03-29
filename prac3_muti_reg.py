# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:39:04 2021

@author: T30518
"""


import numpy as np 
from sklearn.linear_model import LinearRegression 
X = np.array([ [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180] ]) 
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364]) 
reg= LinearRegression() 
reg.fit(X, y) 
#印出迴歸係數,beta值
#這邊因為X有兩個，所以回歸係數也會有兩個
print(reg.coef_) 
# 印出截距 
print(reg.intercept_ ) 
#產出模型後，用想預測的值投入，得442
predicted = np.array([ [10, 110] ]) 
predicted_sales= reg.predict(predicted) 
print("%d" % predicted_sales)
