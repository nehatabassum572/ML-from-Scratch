#!/usr/bin/env python
# coding: utf-8

# In[12]:


#pip install --upgrade numpy>=2


# In[35]:


import numpy as np


# In[36]:


# In df, x is input, y is output
# m is weights
# b is bias
# lr is learning rate 
# epochs is number of iterations


# In[37]:


class LR:
    def __init__(self, lr = 0.01, epochs = 500):
        self.lr = lr
        self.epochs = epochs
        self.m = None
        self.b = None
        
    def fit(self, x, y):
        n_rows, n_features = x.shape
        self.m = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.epochs):
            y_pred = np.dot(x, self.m) + self.b

            m_gradient = (2/n_rows) * np.dot(x.T, (y_pred - y))
            b_gradient = (2/n_rows) * np.sum(y_pred - y)

            self.m = self.m - self.lr * m_gradient
            self.b = self.b - self.lr * b_gradient

    def predict(self, x):
        y_pred = np.dot(x, self.m) + self.b
        return y_pred
        


# In[ ]:
