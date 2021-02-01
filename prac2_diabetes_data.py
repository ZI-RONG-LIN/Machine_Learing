# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:51:09 2021

@author: jxunchen
"""
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#資料集
#載入資料
diabetes = datasets.load_diabetes() 
#印出此資料及的介紹
print(diabetes.DESCR)

#資料分兩部分，一份是訓練，一份是測試
#本身資料就已經有附上標記了，如果是其他資料集則要自己處理標記
#獲取一個特徵，就是取出觀察值而已，只是另一種說法
#diabetes.data，diabetes資料集中的觀察值
diabetes_x_temp = diabetes.data[:, np.newaxis, 2] 
#訓練樣本
#從頭取到最後20筆
diabetes_x_train = diabetes_x_temp[:-20]  
#測試樣本 後20行
#倒數20筆資料
diabetes_x_test = diabetes_x_temp[-20:]    
#訓練標記
#diabetes.target印出資料的標記
diabetes_y_train = diabetes.target[:-20]   
#預測對比標記
diabetes_y_test = diabetes.target[-20:]    

#迴歸訓練及預測
reg = LinearRegression()
#注: 訓練資料集
#用訓練資料去fit測試資料跑出來的模型
reg.fit(diabetes_x_train, diabetes_y_train)  

#係數 殘差平法和 方差得分
#迴歸係數
print ('Coefficients :\n', reg.coef_)
#殘差平方和
print ("Residual sum of square: %.2f" %np.mean((reg.predict(diabetes_x_test) - diabetes_y_test) ** 2))
#這邊應該是判定係數，不是變異數
print ("Coefficient of determination: %.2f" % reg.score(diabetes_x_test, diabetes_y_test))

#繪圖
#標題
plt.title(u'LinearRegression Diabetes')   
#x軸座標
plt.xlabel(u'Attributes')    
#y軸座標             
plt.ylabel(u'Measure of disease')     
    
#點的準確位置
#畫散佈圖
plt.scatter(diabetes_x_test, diabetes_y_test, color = 'black')
#預測結果 直線表示
plt.plot(diabetes_x_test, reg.predict(diabetes_x_test), color='blue', linewidth = 3)
plt.show()