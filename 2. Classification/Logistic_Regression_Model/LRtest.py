import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets 

df = datasets.load_breast_cancer()
X, y = df.data , df.target

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

from LogisticRegression import LR
model_LR = LR(lr = 0.01, epochs = 1000)
model_LR.fit(X_train, y_train)
predictions = model_LR.predict(X_test)

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print('Logistic Regression Accuracy Score = ',accuracy_score(y_test, predictions))

