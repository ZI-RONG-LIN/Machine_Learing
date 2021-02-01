# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:37:32 2021

@author: T30518
"""

#import numpy as np 
#觀察資料
import matplotlib.pyplot as plt
#因為後面的函數格式規定要求，所以要list裡面再包list
#不然如果x只有一個的話，其實可以只用一個list包就好
X = [[6], [8], [10], [14], [18]] 
y = [[7], [9], [13], [17.5], [18]]
plt.figure() 
plt.title('Pizza Price with diameter.') 
plt.xlabel('diameter(inch)') 
plt.ylabel('price($)') 
plt.axis([0, 25, 0, 25]) 
#plt.grid(True) 顯示網格線
plt.grid(True) 
#'k.'為用'黑色'的點顯示
plt.plot(X, y, 'k.') 
plt.show()

###建立模型與評量 

from sklearn.linear_model import LinearRegression
reg= LinearRegression() 
# X and y is the data in previous code. 
reg.fit(X, y) 
print(u'係數', reg.coef_) 
print (u'截距', reg.intercept_) 
#顯示判定係數R^2
print (u'評分函式', reg.score(X, y))
X2 = [[1], [10], [14], [25]] 
y2 = reg.predict(X2) 
print(y2)

###資料預測
X2 = [[1], [10], [14], [25]] 
y2 = reg.predict(X2) 
#繪製線性迴歸圖形
print(y2)  
plt.figure()
#標題  
plt.title(u'PizzaPrice with diameter.')  
#x軸座標 
plt.xlabel(u'diameter') 
#y軸座標              
plt.ylabel(u'price')    
#區間               
plt.axis([0, 25, 0, 25]) 
#顯示網格              
plt.grid(True) 
#繪製訓練資料集散點圖                      
plt.plot(X, y, 'k.')  
#繪製預測資料集直線       
plt.plot(X2, y2, '-g.')              

