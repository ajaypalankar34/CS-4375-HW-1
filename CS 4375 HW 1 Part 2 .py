#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
LinearRegressionModel= linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from math import sqrt


# In[3]:


data = pd.read_excel("ENB2012_data.xlsx")


# In[4]:


data.dropna(0)


# In[24]:


data.head


# In[7]:


data_feature=data.iloc[:,0:8]
data_feature.hist(figsize=(20,10))
plt.show()


# In[10]:


x = data.drop('Y1', axis=1)
y = data.iloc[:,-1]


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
print ('Number of samples in training data:',len(x_train))
print ('Number of samples in validation data:',len(x_test))


# In[12]:


print ('Training a Linear Regression Model..')
reg=LinearRegression().fit(x_train,y_train)


# In[13]:


print('Coefficient Values = ',reg.coef_)


# In[14]:


print('Intercept Value =',reg.intercept_)


# In[15]:


y_train_predict = reg.predict(x_train)


# In[16]:


y_train_predict.shape


# In[19]:


training_error = sqrt(mean_squared_error(y_train, y_train_predict))
print ('Training Error:',training_error)


# In[20]:


y_test_predict = reg.predict(x_test)
y_test_predict.shape


# In[32]:


testing_error = sqrt(mean_squared_error(y_test, y_test_predict))
print ('Testing Error:',testing_error)


# In[28]:


amount=[100,200,300,400,500,614]
training_error_list=[]
testing_error_list=[]
for i in amount:
    x_train_i = x_train[:i]
    y_train_i = y_train[:i]
    reg_i = LinearRegression().fit(x_train_i,y_train_i)
    y_train_i_predict = reg_i.predict(x_train_i)
    training_error_i = sqrt(mean_squared_error(y_train_i, y_train_i_predict))
    print ('Training Error for',i, 'amount of Traning data:',training_error_i)
    y_test_i_predict = reg_i.predict(x_test)
    testing_error_i = sqrt(mean_squared_error(y_test, y_test_i_predict))
    print ('Test Error for',i, 'amount of Test data:',testing_error_i,'\n')
    training_error_list.append(training_error_i)
    testing_error_list.append(testing_error_i)
    
fig, ax = plt.subplots(figsize=(10, 6))
my_xticks = ['100', '200', '300', '400', '500','614']
p1 = ax.plot(my_xticks,training_error_list, marker="^", markersize=12, color='b')
p2 = ax.plot(my_xticks,testing_error_list,marker='.',markersize=12, color='y')
ax.set_title('Error Rates vs. Number of Training Examples',fontsize=20)
ax.set_xlabel('Number of Training Examples')
ax.set_ylabel('Error Rates')
ax.legend((p1[0], p2[0]), ('Training Error', 'Test Error'),loc='center left', bbox_to_anchor=(1, 0.5))
ax.autoscale_view()


# In[26]:


r2_score(y_test,y_test_predict)


# In[27]:


r2_score(y_train,y_train_predict)


# In[30]:


mean_squared_error(y_test,y_test_predict)


# In[31]:


mean_squared_error(y_train, y_train_predict)


# In[ ]:




