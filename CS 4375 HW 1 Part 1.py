#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd 
import numpy as np
from numpy.linalg import inv


# In[11]:


data = pd.read_excel("ENB2012_data.xlsx")


# In[12]:


colums = (data.columns[0])


# In[13]:


max= [data[c].max() for c in data.columns] 
min= [data[c].min() for c in data.columns]


# In[14]:


i=0
for c in data.columns:
    while(i<len(data.columns)): 
        data[c]=(data[c]-min[i])/(max[i]-min[i])
        i=i+1
        break


# In[15]:


arr = data.values
x_train=[]
y1=[]
y2=[]
a=data.shape
for i in range(a[0]):                      
    x_train.append((arr[i][:-2]).tolist())
    y2.append(arr[i][-1])
    y1.append(arr[i][-2])


# In[16]:


m=np.ones((768,1))
b=np.matrix(x_train)
b=np.concatenate((m,b),axis=1)     
d=b.T
e=np.linalg.inv(np.matmul(d,b))
y1=np.matrix(y1)
y1=y1.T
y2=np.matrix(y2)
y2=y2.T
f=np.matmul(e,d)
c1=np.matmul(f,y1)
c2=np.matmul(f,y2)


# In[17]:


x_test=[[1],]
for j in range (8):
    inp=[float(input("Enter Value:"))]
    x_test.append(inp)
for i in range(8):
    x_test[i+1][0]=(x_test[i+1][0]-min[i])/(max[i]-min[i])
x_test=np.matrix(x_test)
Y1=np.matmul(c1.T,x_test)
Y2=np.matmul(c2.T,x_test)


# In[27]:


Y1_predicted = (Y1*(max[-2]-min[-2])+min[-2])
Y2_predicted = ((Y2*(max[-1]-min[-1]))+min[-1])


# In[28]:


print(Y1_predicted)
print(Y2_predicted)


# In[30]:


x_test=[[1],]
for j in range (8):
    inp=[float(input("Enter Value:"))]
    x_test.append(inp)
for i in range(8):
    x_test[i+1][0]=(x_test[i+1][0]-min[i])/(max[i]-min[i])
x_test=np.matrix(x_test)
Y1=np.matmul(c1.T,x_test)
Y2=np.matmul(c2.T,x_test)


# In[ ]:


Y1_predicted = (Y1*(max[-2]-min[-2])+min[-2])
Y2_predicted = ((Y2*(max[-1]-min[-1]))+min[-1])


# In[ ]:


print(Y1_predicted)
print(Y2_predicted)


# In[ ]:


x_test=[[1],]
for j in range (8):
    inp=[float(input("Enter Value:"))]
    x_test.append(inp)
for i in range(8):
    x_test[i+1][0]=(x_test[i+1][0]-min[i])/(max[i]-min[i])
x_test=np.matrix(x_test)
Y1=np.matmul(c1.T,x_test)
Y2=np.matmul(c2.T,x_test)

