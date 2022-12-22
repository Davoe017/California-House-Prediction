#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import math


# In[2]:


Cali_df = pd.read_excel(r'C:\Users\HP\Desktop\California_housing.xlsx')


# In[3]:


Cali_df.head()


# In[4]:


Cali_df.columns


# In[5]:


Cali_df.isnull().sum()


# In[6]:


Cali_df['total_bedrooms'] = Cali_df['total_bedrooms'].fillna(Cali_df['total_bedrooms'].mean())
Cali_df.isnull().sum()


# In[7]:


Label_Enc = LabelEncoder()
Cali_df['ocean_proximity'] = Label_Enc.fit_transform(Cali_df['ocean_proximity'])


# In[17]:


x_input=['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 
        'population', 'households', 'median_income','ocean_proximity']
x_data = Cali_df[x_input]
y_data = Cali_df['median_house_value']


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)


# In[21]:


scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.fit_transform(x_test)


# In[22]:


Lin_Reg = LinearRegression()
Lin_Reg.fit(scaled_x_train, y_train)


# In[23]:


Lin_Reg_ypred = Lin_Reg.predict(scaled_x_test)


# In[26]:


print(math.sqrt(mean_squared_error(y_test, Lin_Reg_ypred)))


# In[27]:


Dec_Tree_Reg = DecisionTreeRegressor()
Dec_Tree = Dec_Tree_Reg.fit(scaled_x_train, y_train)


# In[28]:


Dec_Tree_ypred = Dec_Tree.predict(scaled_x_test)


# In[29]:


print(math.sqrt(mean_squared_error(y_test, Dec_Tree_ypred)))


# In[32]:


Rand_Forest = RandomForestRegressor()
Rand_For_M = Rand_Forest.fit(scaled_x_train, y_train)


# In[33]:


Rand_For_ypred = Rand_For_M.predict(scaled_x_test)


# In[34]:


print(math.sqrt(mean_squared_error(y_test, Rand_For_ypred)))


# In[40]:


x_data_new = np.array(Cali_df['median_income']).reshape(-1,1)
y_data_new = Cali_df['median_house_value']

scaler_new = StandardScaler()
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_data_new, y_data_new, test_size=0.2, random_state=0)
scaled_x_train_new = scaler_new.fit_transform(x_train_new)
scaled_x_test_new = scaler_new.fit_transform(x_test_new)


# In[44]:


Lin_Reg_new = LinearRegression()
Lin_Reg_new  = Lin_Reg_new.fit(scaled_x_train_new, y_train_new)
Lin_Reg_new_ypred = Lin_Reg_new.predict(scaled_x_test_new)


# In[47]:


plt.scatter(scaled_x_train_new,y_train_new,color='green')
plt.plot(scaled_x_train_new, Lin_Reg_new.predict(scaled_x_train_new),color='red')
plt.title('Median House Price Prediction')
plt.xlabel('Median income')                
plt.ylabel('House price')
plt.show()


# In[48]:


plt.scatter(scaled_x_test_new, y_test_new, color='blue')
plt.plot(scaled_x_test_new, Lin_Reg_new.predict(scaled_x_test_new), color='yellow')
plt.title('Median House Price Prediction')
plt.xlabel('Median income')
plt.ylabel('House price')
plt.show()


# In[ ]:




