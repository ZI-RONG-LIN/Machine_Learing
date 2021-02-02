#!/usr/bin/env python
# coding: utf-8

# # 3.4 Get the Data 
# 
# * The full Jupyter notebook is available at https://github.com/ageron/handson-ml
# * https://github.com/ageron/handson-ml/tree/master/datasets/insurance

# In[4]:


import pandas as pd # 輸入
#匯入csv
insurance = pd.read_csv("C:/Users/T30518/Desktop/0202_ML/data/data/lec03-insurance.csv") # 資料夾 data 下 lec03-insurance.csv
insurance.head(2) 


# In[10]:


#輸出資料集中前五筆資料
insurance.head() 


# In[11]:


#輸出資料表的屬性
insurance.info()


# In[31]:


#輸出資料總筆數、欄位數
insurance.shape


# In[32]:


#輸出年齡欄位的資料
insurance.age


# In[33]:


#輸出年齡欄位的資料
insurance['age']


# In[13]:


#輸出年齡欄位的資料
#這個是直接抓第1欄的數據出來，就不用指名age欄位了
insurance.iloc[:,0] 


# In[14]:


#取出前兩筆數據
insurance.head(2)


# In[15]:


#讀取第二筆資料的所有資訊
insurance.iloc[1]


# In[37]:


#取出前兩筆數據
insurance.head(2)


# In[17]:


#insurance.iloc[0, 5]是指取第1筆資料的第6欄
#insurance['region'][0]取region欄位的第1筆資料
insurance.iloc[0, 5], insurance['region'][0]


# In[39]:


#取出前兩筆數據
insurance.head(2)


# In[40]:


#取第一筆資料第6欄以後的數據
insurance.iloc[0, 5:] 


# In[41]:


insurance.head(2)


# In[45]:


#輸出第2筆資料，第2&3欄位的數據
insurance.iloc[1, 1:3] 


# In[18]:


#輸出描述性統計的數據
insurance.describe()


# In[47]:


#insurance.age.mean()計算年齡欄位的平均
#取出描述性統計數據結果中，第2筆數據的第1欄資料結果
insurance.age.mean(), insurance.describe().iloc[1, 0]


# In[48]:


#取出描述性統計數據結果中，第5~7筆數據的第1欄資料結果
insurance.describe().iloc[4:7, 0]


# # 3.4.2 Nominal data (名目資料)

# In[49]:


insurance.head(2)


# In[51]:


#看insurance中，region欄位的描述性統計資料
#top 跟 freq只會顯示眾數的資料
#其他的話要用value_count函數做計算
insurance.region.describe()


# In[59]:


#可以參考[60]程式碼的結果
364/1338, insurance.region.describe()[3] / insurance.region.describe()[0]


# In[60]:


insurance.describe(include = 'all')


# In[19]:


#輸出region值，如果沒有指定為類別的話不會統整成一組一組
c = insurance.region.values
c


# In[61]:


#指定region為類別資料
c = insurance.region.astype('category').values
c


# In[63]:


#計算region的各類別的頻率
c.value_counts()


# In[65]:


insurance.region.value_counts()


# # 3.4.3 NumPy 

# In[66]:


import numpy as np
A = np.array([2, 2, 4, 80])
np.mean(A), np.median(A) #平均、中位數


# In[20]:


import numpy as np
#scipy就包含pandas跟numpy
from scipy import stats # scipy also include pandas and NumPy
A = np.array([2, 2, 2, 4, 80])
#印出A的眾數，
print(stats.mode(A))
#mode=array([2]), count=array([3] 2出現了3次的意思
# ModeResult(mode=array([2]), count=array([3]))
print(stats.mode(A)[0][0], stats.mode(A)[1][0]) # 2, 3 


# In[68]:


B = np.array([2, 2, 4, 8])
#計算變異數、標準差、最小、最大值
np.var(B), np.std(B), [np.min(B), np.max(B)]


# ## 2-dim array 

# In[69]:


import numpy as np
b = np.array(([4, 5, 1], [6, 2, 7]))
b


# In[70]:


#b為幾*幾的矩陣
#b.shape[0] 列的數量
#b.shape[1] 行的數量
b.shape, b.shape[0], b.shape[1] # 形狀


# In[73]:


b


# In[71]:


b[:, 2]


# In[72]:


b[1]


# In[74]:


b[:, 0:2] 


# In[75]:


#建立2*3的0矩陣
np.zeros((2,3)) 


# In[76]:


np.eye(3,3) # identity matrix (單位矩陣)


# ## 3.4.4 Create a Test Set

# In[24]:


import numpy as np
#隨機洗牌
#每次結果都不一樣
np.random.permutation(5)


# In[78]:


import numpy as np
np.random.permutation(5)


# In[32]:


#如果隨機種子一樣的話，每次洗完的結果都會一樣

np.random.seed(25)
shuffled_indices = np.random.permutation(5)
print(shuffled_indices, shuffled_indices[:3], shuffled_indices[3:])


# In[30]:


np.random.seed(42)
shuffled_indices = np.random.permutation(5)
print(shuffled_indices, shuffled_indices[:3], shuffled_indices[3:])


# In[35]:


#自己寫函數洗牌
import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    # to make this notebook's output identical at every run
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[36]:


train_set, test_set = split_train_test(insurance, 0.2)
print(len(train_set), "train +", len(test_set), "test", ' = total ',  len(train_set) + len(test_set))


# In[83]:


train_set.info()


# In[84]:


test_set.info()


# In[85]:


train_set.head(2)


# ### another approach

# In[37]:


#取前兩筆數據的所有欄位
insurance.iloc[0:2, :]


# In[38]:


#把dataset分成訓練集跟測試集
from sklearn.model_selection import train_test_split 
# Split dataset
#X想看的變數，insurance.iloc[:, 0:6]取全部資料的第0~6欄的資料
#y想看的結果，insurance.iloc[:, 6]取全部資料的第7欄的資料
#random_state = 42固定取的人，才不會說X取的人跟Y取的人不一樣
X_train, X_test, Y_train, Y_test = train_test_split(insurance.iloc[:, 0:6], insurance.iloc[:, 6], test_size=0.2, random_state = 42)


# In[39]:


train_set.head()


# In[98]:


X_train.head(2)


# In[40]:


Y_train.head(2)


# In[ ]:




