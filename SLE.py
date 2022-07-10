#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# ## Importing Data set

# In[2]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
data.head()


# ## Handling Nan values

# In[3]:


Hours_median = data.Hours.median()
Hours_median = math.floor(Hours_median)
data.Hours = data.Hours.fillna(Hours_median)
data.head()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data.Hours,data.Scores,marker = '+',color = 'red')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores")
plt.show()


# ## Preparing the data

# In[5]:


x = data.iloc[:,:-1].values #selects all the column except last column
y = data.iloc[:,1].values   #selects only last column


# ## Spliting up into train-set and test-test

# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# ## Training simple linear model

# In[7]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)


# ## visualising training set

# In[8]:


plt.scatter(x_train,y_train,color = 'red',marker = '+')
plt.plot(x_train,reg.predict(x_train))
plt.title("Hours vs Scores")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# ## visualising test set

# In[9]:


plt.scatter(x_test,y_test,color = 'red',marker = '+')
plt.plot(x_train,reg.predict(x_train))
plt.title("Hours vs Scores")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# ## Making Predictions

# In[10]:


y_pred = reg.predict(x_test)
df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
df


# In[11]:


hours = [[9.25]]
own_pred = reg.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaluating the Model

# In[12]:


from sklearn import metrics
print('Mean Absolute Error: ',metrics.mean_absolute_error(y_test,y_pred))

