# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:15:35 2021

@author: T30518
"""

#IRIS datatset
#seaborn 畫圖套件，有提供載入資料的功能
#比較不同品種的花之間的屬性
import seaborn as sns
iris = sns.load_dataset('iris') 
iris.head() 
sns.set() 
sns.pairplot(iris, hue='species', size=3)
