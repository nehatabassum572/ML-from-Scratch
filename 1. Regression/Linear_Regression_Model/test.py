

# In[1]:


#pip install --upgrade numpy>=2


# In[12]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets


# In[16]:


# LOAD DATASET 
x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)


# In[19]:


fig1 = plt.figure(figsize=(8,6))
plt.scatter(x, y, color = "b", marker = "o", s = 30)
plt.show()


# In[20]:


# SPLIT THE DATASET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


# In[23]:


from LinearRegression import LR

model_LR = LR()
model_LR.fit(x_train,y_train)
y_predictions = model_LR.predict(x_test)

def mse(y_test, predictions):
    return np.mean((y_test - y_predictions)**2)

mse = mse(y_test, y_predictions)
print(mse)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_predictions) * 100)


# In[ ]:


y_pred_line = model_LR.predict(x)
cmap = plt.get_cmap('viridis')
fig2 = plt.figure(figsize=(8,6))
m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(x, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()

