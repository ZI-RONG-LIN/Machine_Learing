#!/usr/bin/env python
# coding: utf-8

# ## From lecture 3

# In[1]:


import pandas as pd # 輸入
import numpy as np
insurance = pd.read_csv("C:/Users/user/Desktop/0202_ML/data/data/lec03-insurance.csv") # 資料夾 data 下 insurance.csv

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # to make this notebook's output identical at every run
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(insurance, 0.2)
train_set.info()


# In[2]:


test_set.info()


# # 4.1 Discover (發現) and Visualize (視覺化) the Data to Gain Insights (洞察)

# In[3]:


train_set.head(2)


# In[536]:


insurance = train_set.copy()
insurance.head(2)


# # 4.1.1 Single variable (單變量)

# In[8]:


#畫直方圖
import matplotlib.pyplot as plt
insurance.charges.hist(bins = 5, figsize=(20,15))
#建立路徑，儲存圖檔
plt.savefig("C:/Users/user/Desktop/0202_ML/data/data//charges")
plt.show()


# In[538]:


insurance.charges.hist(bins = 10)
plt.show()


# In[539]:


#最大最小值
insurance.charges.min(), insurance.charges.max()


# In[540]:


#建立charges的描述性統計
insurance.charges.describe()


# In[ ]:





# In[ ]:





# # 4.1.2 Looking for Correlations (尋找相關性)

# In[541]:


#計算insurance裡面各特徵之間的相關性
corr_matrix = insurance.corr()
corr_matrix 


# In[542]:


#顯示charges和其他屬性之間的相關性，並以降序呈現
corr_matrix["charges"].sort_values(ascending=False)


# In[543]:


#顯示charges和其他屬性之間的相關性，並以升序呈現
corr_matrix["charges"].sort_values()


# ## Histograms (直方圖)

# In[544]:


import matplotlib.pyplot as plt
#只畫charge的直方圖
insurance.charges.hist(bins=50, figsize=(20,15))
plt.show()


# In[9]:


# %matplotlib inline
import matplotlib.pyplot as plt
#畫出insurance所有特徵的直方圖
insurance.hist(bins=50, figsize=(20,15))
# plt.savefig("data/insurance hist")
plt.show()


# In[546]:


#畫age跟charges的散布圖，alpha是透明度，0.0(透明)->1.0(不透明)
insurance.plot(kind="scatter", x="age", y="charges", alpha = 0.8)
#前面兩個是x軸的上下界範圍，後兩個是y的上下界範圍
plt.axis([18, 64, 1000, 65000])
#建立圖標題
plt.title('alpha = 0.8')
#存圖
plt.savefig("data/age_vs_charges_scatterplot_8")
plt.show()


# In[547]:


insurance.plot(kind="scatter", x="age", y="charges", alpha= 0.2)
plt.axis([18, 64, 1000, 65000])
plt.title('alpha = 0.2')
plt.savefig("data/age_vs_charges_scatterplot_2")
plt.show()


# In[548]:


insurance.head()


# In[549]:


#scatter_matrix 散佈圖矩陣
from pandas.plotting  import scatter_matrix
#有n個特徵就產出n*n個圖，左上至右下的斜對角為該特徵的直方圖
attributes = ["age", "bmi", "children" , "charges"]
#設定圖的大小
scatter_matrix(insurance[attributes], figsize=(11, 8))
plt.savefig("data/scatter_matrix_plot")
plt.show()


# # 4.4 Prepare the Data for Machine Learning Algorithms (機器學習演算法)

# In[550]:


insurance.head()


# In[551]:


insurance.describe()


# In[10]:


#把charges這欄去除，axis=1為去除欄，axis=0為去除行
insurance = train_set.drop("charges", axis=1)
#複製charges這欄作為label
insurance_labels = train_set["charges"].copy()


# In[553]:


insurance.info()


# In[554]:


insurance.head(2)


# In[555]:


insurance_labels.describe()


# In[556]:


insurance.info()


# In[557]:


insurance_labels.head(2)


# # 4.2.1 Dealing with missing Data

# In[12]:


import pandas as pd
insurance5 = pd.read_excel('C:/Users/user/Desktop/0202_ML/data/data/lec03-insurance-5.xlsx') 
insurance5.head()


# ## (1) Delete
# 
# * https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html

# In[13]:


#把有NA值的數據去除
insurance5.dropna()


# In[14]:


insurance5.dropna().info()


# * ‘all’ : If all values are NA, drop that row or column.

# In[15]:


#如果所有值都是NA時，刪除該筆數據
insurance5.dropna(how = 'all')


# ## (2) Replace with summary

# In[16]:


#印出bmi的數據
insurance5.bmi 


# In[563]:


#計算bmi的平均
insurance5.bmi.mean()


# In[564]:


#計算性別的眾數為何
insurance5.sex.mode()


# In[25]:


#取出眾數矩陣的第一筆資料
insurance5.sex.mode()[0]


# google fillna multiple columns
# 
# * https://stackoverflow.com/questions/34913590/fillna-in-multiple-columns-in-place-in-python-pandas
# * https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.dtype.kind.html
# * https://www.w3schools.com/python/python_lambda.asp

# In[566]:


insurance5.head()


# In[27]:


insurance5.fillna(insurance5.mode())


# In[26]:


#用該特徵的眾數取代NA值
insurance5.fillna(insurance5.mode().iloc[0])


# In[568]:


insurance5.head()


# In[28]:


#沒取代成功，因為先用眾數取代了，正確如569的程式碼
insurance5.fillna(insurance5.mode().iloc[0]).fillna(insurance5.median())


# In[569]:


#數值資料的NA先用中位數取代後，類別資料再用眾數取代
insurance5.fillna(insurance5.median()).fillna(insurance5.mode().iloc[0])


# ## (3) Random replace

# In[29]:


import pandas as pd # 輸入
insurance = pd.read_csv('C:/Users/user/Desktop/0202_ML/data/data/lec03-insurance.csv')  # 資料夾 data 下 housing.csv
insurance.bmi.describe()


# In[571]:


insurance.bmi.min(), insurance.bmi.max()


# In[31]:


#取道小數點後一位
insurance.bmi.min().round(1), insurance.bmi.max().round(1)


# In[35]:


#隨機替換
#再最大值與最小值之間隨機取一個數，重複3次取平均作為取代值
import random
random.randrange(insurance.bmi.min().round(), insurance.bmi.max().round())


# In[574]:


insurance5.head()


# google random element fillna
# 
# * https://stackoverflow.com/questions/47497466/python-fill-na-in-pandas-column-with-random-elements-from-a-list
# * https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html

# In[36]:


#把region的值作為類別資料，並把全部值輸出
insurance.region.astype('category').values


# In[575]:


#把region的值作為類別資料，只顯示類別
insurance.region.astype('category').values.categories


# In[576]:


#把region的值作為類別資料，只顯示第2個類別
insurance.region.astype('category').values.categories[1]


# In[577]:


region_cat = insurance.region.astype('category').values.categories


# In[578]:


import numpy as np
np.random.choice(region_cat), np.random.choice(region_cat), np.random.choice(region_cat)


# # 4.2.2 Managing categorical data (類別資料)
# 
# * https://scikit-learn.org/stable/modules/preprocessing.html: The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators. 

# In[579]:


insurance_labels.head()


# In[580]:


insurance.head()


# In[581]:


insurance.region.astype('category').values.categories


# In[582]:


insurance.region[0] 


# In[583]:


insurance5.head()


# In[584]:


insurance5.region[0] # The first element


# In[585]:


#標籤編碼LabelEncoder，類別轉數值，從0開始編到n
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
insurance_cat = insurance["region"]
insurance_cat_encoded = encoder.fit_transform(insurance_cat)
print(encoder.classes_)
print(insurance_cat_encoded)


# In[586]:


#獨熱編碼，等於是創4個類別，用0、1表示是不是屬於該類別，像[587]程式碼
from sklearn.，等於是創4個類別，用0、1表示是不是屬於該類別preprocessing import LabelBinarizer
#LabelBinarizer標籤二元化
encoder = LabelBinarizer()
insurance_cat = insurance["region"]
insurance_cat_1hot = encoder.fit_transform(insurance_cat)
insurance_cat_1hot


# In[587]:


insurance_cat_1hot_df = pd.DataFrame(insurance_cat_1hot, columns = encoder.classes_)
insurance_cat_1hot_df.head()


# # 4.4.3 Data scaling and normalization (資料縮放與正規化)
# 
# * 以 column 為依據

# In[38]:


import numpy as np
x = np.array([[1, 2],[2, 6], [6, 1]])
x


# In[39]:


#極值標準化
from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()
scaled_data = ss.fit_transform(x)
scaled_data


# In[590]:


#Z-分數標準化
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
scaled_data = ss.fit_transform(x)
scaled_data


# # 4.2.4 Size of Data Frame

# In[51]:


import pandas as pd # 輸入
import numpy as np
insurance = pd.read_csv("C:/Users/user/Desktop/0202_ML/data/data/lec03-insurance.csv") # 資料夾 data 下 insurance.csv
# 定義一個分測試集跟訓練集的function
# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # to make this notebook's output identical at every run
    np.random.seed(42)  
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(insurance, 0.2)
insurance = train_set.drop("charges", axis=1)
insurance_labels = train_set["charges"].copy()


# In[592]:


insurance.head()


# In[49]:


insurance.info()


# In[53]:


insurance = insurance.reset_index(drop=True)
insurance.head() 


# ## Replace with summary  (替換為摘要) 

# In[595]:


insurance = insurance.fillna(insurance.median()).fillna(insurance.mode().iloc[0])
insurance.info()


# ## 數值資料 3 個
# 
# * google pandas drop many columns
#     * https://stackoverflow.com/questions/28538536/deleting-multiple-columns-based-on-column-names-in-pandas

# In[596]:


insurance_num = insurance.drop(['sex', 'smoker', 'region'], axis=1)
insurance_num.head()


# # Handling Text and Categorical Attributes
# 
# * 2 classes
#     * https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes

# In[597]:


insurance.head(10)


# In[598]:


from sklearn.preprocessing import LabelBinarizer
import numpy as np

encoder = LabelBinarizer()
sex_ohe = encoder.fit_transform(insurance["sex"])
sex_ohe  


# In[599]:


encoder.classes_


# In[600]:


from sklearn.preprocessing import LabelBinarizer
import numpy as np

encoder = LabelBinarizer()
sex_ohe = encoder.fit_transform(insurance["sex"])
sex_ohe = np.hstack((1 - sex_ohe, sex_ohe))
sex_df = pd.DataFrame(sex_ohe, columns = encoder.classes_)
sex_df.head(10)


# In[601]:


insurance.head(2)


# In[602]:


#hstack水平堆疊矩陣
smoker_ohe = encoder.fit_transform(insurance["smoker"])
smoker_ohe = np.hstack((1 - smoker_ohe, smoker_ohe))
smoker_df = pd.DataFrame(smoker_ohe, columns = encoder.classes_)
smoker_df.head(2)


# In[603]:


insurance.head(2)


# In[604]:


region_ohe = encoder.fit_transform(insurance["region"])
region_df = pd.DataFrame(region_ohe, columns = encoder.classes_)
region_df.head(2)


# In[605]:


insurance.head()


# In[606]:


insurance_df = pd.concat([insurance_num, sex_df, smoker_df, region_df], axis=1)
insurance_df.head()


# In[607]:


insurance_labels = insurance_labels.reset_index(drop=True)
insurance_labels.head() 


# Better approach
# 
# * https://github.com/pandas-dev/pandas/issues/12042

# In[608]:


cat_df = pd.concat([insurance["sex"], insurance["smoker"], insurance["region"]], axis=1)
cat_df.head()


# In[609]:


#建dummy
pd.get_dummies(cat_df).head()


# In[610]:


all_data = pd.concat([insurance_df, insurance_labels], axis=1)


# In[56]:


corr_matrix = insurance.corr()
corr_matrix 


# In[611]:


all_data.corr()


# In[612]:


all_data.corr()["charges"].sort_values(ascending=False)


# # 4.3 Select and Train a Model
# 
# * Linear Regression for training set
# 
#     * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#         * If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the l2-norm.

# In[613]:


from sklearn.linear_model import LinearRegression
# Create a linear regressor instance
lr = LinearRegression(normalize=True)
# Train the model
lr.fit(insurance_df, insurance_labels)


# infor 
# 
# * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#     * Returns the coefficient of determination R^2 of the prediction

# In[614]:


print( "Score {:.4f}".format(lr.score(insurance_df, insurance_labels)) ) 


# ### Linear Regression Equation

# In[615]:


print('y = %.3f '% lr.intercept_)
for i, c in enumerate(lr.coef_):
    print('%.3f '% c, insurance_df.columns.values[i])


# In[616]:


some_data = insurance_df.iloc[:4] # 4 個例子
some_labels = insurance_labels.iloc[:4]
print("Predictions:\t", lr.predict(some_data))
print("Labels:\t\t", list(some_labels))


# ## The 4th Sample

# In[617]:


insurance_df.iloc[3]


# In[618]:


insurance_labels.iloc[3]


# In[619]:


lr.predict(insurance_df.iloc[3:5])


# In[620]:


predicted_y = lr.intercept_

for i, c in enumerate(lr.coef_):
    predicted_y += c * insurance_df.iloc[3][i]
    
print('predicted y = %.3f '% predicted_y)


# In[621]:


insurance_all = pd.concat([insurance_df, insurance_labels], axis=1)
insurance_all.head()


# In[622]:


insurance_all[insurance_all.age == 52]


# In[623]:


insurance_all[(insurance_all.age == 52) & (insurance_all.female == 1) ]


# In[624]:


insurance_all[(insurance_all.age == 52) & (insurance_all.female == 1) & (insurance_all.children == 0) ]


# In[625]:


some_data = insurance_df.iloc[721:724]  
some_labels = insurance_labels.iloc[721:724]
print("Predictions:\t", lr.predict(some_data))
print("Labels:\t\t", list(some_labels))


# In[ ]:





# In[626]:


lr.predict(insurance_df).min(), lr.predict(insurance_df).max()


# In[627]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr.predict(insurance_df) )))


# In[628]:


insurance_labels.min(), insurance_labels.max()


# In[629]:


lr.predict(insurance_df)


# In[630]:


insurance_df.describe()


# ## 4.3.2 Multicollinearity (多重共線性) 
# 
# * https://www.linkedin.com/pulse/super-simple-machine-learning-multiple-linear-regression-low/

# ### insurance_new_df

# In[631]:


cat_df.head(2)


# In[632]:


cat_df.index


# In[633]:


new_df = pd.DataFrame(index = cat_df.index)
for i in cat_df:
    new_df = new_df.join(pd.get_dummies(cat_df[i]).iloc[:, 1:])

new_df.head()


# In[634]:


insurance_new_df = pd.concat([insurance_num, new_df], axis=1)
insurance_new_df.head()


# In[635]:


lr2 = LinearRegression(normalize=True)
# Train the model
lr2.fit(insurance_new_df, insurance_labels)
print( "Score {:.4f}".format(lr2.score(insurance_new_df, insurance_labels)) ) 


# In[636]:


print('y = %.3f '% lr2.intercept_)
for i, c in enumerate(lr2.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])


# ## 手動 StandardScaler , normalize= False 

# In[637]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaled_data2 = ss.fit_transform(insurance_new_df)

lr5 = LinearRegression(normalize= False)
lr5.fit(scaled_data2, insurance_labels)
print('y = %.3f '% lr5.intercept_)
for i, c in enumerate(lr5.coef_):
    print('%.3f '% c, insurance_new_df.columns.values[i])


# # 4.3.2 Evaluating model performance
# 
# * google significance linear regression python
#     * https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression/27928411

# In[638]:


from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
scaled_data2 = ss2.fit_transform(insurance_new_df)


# In[639]:


import statsmodels.api as sm
#from scipy import stats

X2 = sm.add_constant(scaled_data2)
est = sm.OLS(insurance_labels, X2).fit()
print(est.summary())


# # 4.3.3 Backward selection (向後選擇) 

# In[640]:


insurance_df.head()


# In[641]:


insurance_back = insurance_df.drop(['female', 'male', 'no', 'northeast', 'northwest', 'southeast', 'southwest'], axis=1)
insurance_back.head()


# In[642]:


from sklearn.preprocessing import StandardScaler
ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back)
lr6 = LinearRegression()
# Train the model
lr6.fit(scaled_data6, insurance_labels)
print( "Score {:.4f}".format(lr6.score(scaled_data6, insurance_labels)) ) 
print('y = %.3f '% lr6.intercept_)
for i, c in enumerate(lr6.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])


# In[643]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr6.predict(scaled_data6) )))


# In[644]:


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

ss4 = StandardScaler()
scaled_data4 = ss.fit_transform(insurance_back)


X4 = sm.add_constant(scaled_data4)
est = sm.OLS(insurance_labels, X4)
est2 = est.fit()
print(est2.summary())


# In[645]:


est2.params


# In[646]:


est2.params[2]


# In[647]:


insurance_back.iloc[3]


# In[648]:


predicted_y = est2.params[0]

for i, c in enumerate(est2.params):
    if i == 0:
        predicted_y = est2.params[0]
    else: 
        predicted_y += c * insurance_back.iloc[3][i-1]
    
print('predicted y = %.3f '% predicted_y)


# In[649]:


est2.params[0] + insurance_back.iloc[3][0] * est2.params[1] + insurance_back.iloc[3][1] * est2.params[2] 


# # coefficients are different by using 2 methods
# 
# * solve the normal equation $\theta = (X^T X)^{-1} X^T y$ in lecture 5

# In[650]:


X = insurance_back.values
y = insurance_labels.values
scaled_data = ss.fit_transform(X)

X_b = np.c_[np.ones((insurance_back.shape[0], 1)), scaled_data]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
 
theta_best


# In[651]:


import numpy
numpy.linalg.eig(X_b.T.dot(X_b))


# In[652]:


X = insurance_new_df.values
y = insurance_labels.values
scaled_data = ss.fit_transform(X)

X_b2 = np.c_[np.ones((insurance_new_df.shape[0], 1)), scaled_data]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b2.T.dot(X_b2)).dot(X_b2.T).dot(y)
 
theta_best


# In[653]:


import numpy
numpy.linalg.eig(X_b2.T.dot(X_b2))


# # 4.3.4 Improving model performance 

# In[654]:


insurance_back.head()


# In[655]:


insurance_back['age2'] = insurance_back['age'] ** 2
insurance_back['bmi30_smoker'] =  (insurance_back['bmi'] > 30) * insurance_back['yes']
insurance_back.head(20)


# In[656]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
ss3 = StandardScaler()
scaled_data3 = ss3.fit_transform(insurance_back)

# Create a linear regressor instance
lr3 = LinearRegression()
# Train the model
lr3.fit(scaled_data3, insurance_labels)
print( "Score {:.4f}".format(lr3.score(scaled_data3, insurance_labels)) )


# In[657]:


print('y = %.3f '% lr3.intercept_)
for i, c in enumerate(lr3.coef_):
    print('%.3f '% c, insurance_back.columns.values[i])


# In[658]:


lr3.predict(insurance_back).min(), lr3.predict(insurance_back).max() 


# In[698]:


from sklearn.metrics import mean_squared_error
print(np.sqrt( mean_squared_error(insurance_labels, lr3.predict(scaled_data3) ))) 


# In[660]:


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
# from scipy import stats

ss5 = StandardScaler()
scaled_data5 = ss5.fit_transform(insurance_back)

X5 = sm.add_constant(scaled_data5)
est = sm.OLS(insurance_labels, X5).fit()
print(est.summary())


# ## statistically nonsignificant (不顯著)

# In[661]:


insurance_back.head()


# In[662]:


insurance_back1 = insurance_back.drop(['age'], axis=1)
insurance_back1.head(2)


# In[665]:


ss6 = StandardScaler()
scaled_data6 = ss6.fit_transform(insurance_back1)


X6 = sm.add_constant(scaled_data6)
est = sm.OLS(insurance_labels, X6)
est2 = est.fit()
print(est2.summary())


# In[666]:


insurance_back2 = insurance_back1.drop(['bmi'], axis=1)
print(insurance_back2.head(2))
ss7 = StandardScaler()
scaled_data7= ss7.fit_transform(insurance_back2)


X7 = sm.add_constant(scaled_data7)
est = sm.OLS(insurance_labels, X7)
est2 = est.fit()
print(est2.summary())


# In[678]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
ss8 = StandardScaler()
scaled_data8 = ss3.fit_transform(insurance_back2)

# Create a linear regressor instance
lr8 = LinearRegression()
# Train the model
lr8.fit(scaled_data8, insurance_labels)
print( "Score {:.4f}".format(lr8.score(scaled_data8, insurance_labels)) )
print('y = %.3f '% lr8.intercept_)
for i, c in enumerate(lr8.coef_):
    print('%.3f '% c, insurance_back1.columns.values[i])


# ## test set

# In[680]:


test_set.info()


# In[681]:


test_set.head(2)


# In[682]:


test_set['yes'] = (test_set["smoker"] == "yes").astype(int)
test_set.head(2)


# In[683]:


test_set['bmi30_smoker'] =  (test_set['bmi'] > 30) * test_set['yes']
test_set['age2'] = test_set['age'] ** 2
test_set.head(2)


# In[684]:


insurance_back2.head(2)


# In[685]:


test_set_df = pd.concat([test_set["children"], test_set["yes"], test_set["age2"], test_set["bmi30_smoker"]], axis=1)
test_set_df.head()


# In[686]:


test_set_df = test_set_df.reset_index(drop=True)
test_set_df.head()


# In[687]:


insurance_test_labels = test_set["charges"].copy()
insurance_test_labels = insurance_test_labels.reset_index(drop=True)
insurance_test_labels.head() 


# In[688]:


est2.predict(sm.add_constant(test_set_df))


# In[696]:


from sklearn.preprocessing import StandardScaler
ss8 = StandardScaler()
scaled_data8 = ss8.fit_transform(test_set_df)

print('Score %.3f' % lr8.score(scaled_data8, insurance_test_labels))


# In[699]:


from sklearn.metrics import mean_squared_error
# print(np.sqrt( mean_squared_error(insurance_labels, lr8.predict(insurance_df)) ))
print(np.sqrt( mean_squared_error(insurance_test_labels, lr8.predict(scaled_data8)) ))


# In[700]:


insurance_test_labels.min(), insurance_test_labels.max()


# In[707]:


lr8.predict(scaled_data8).min(), lr8.predict(scaled_data8).max()

