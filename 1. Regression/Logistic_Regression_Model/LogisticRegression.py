import numpy as np 

class LR:
    def __init__(self, epochs = 500, lr = 0.01 ):
        self.lr = lr 
        self.epochs = epochs 
        self.m = None 
        self.b = None 


    def fit(self, X , y):
        n_rows , n_features = X.shape 
        self.m = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            LR_model = np.dot(X, self.m) + self.b
            y_pred = self._sigmoid(LR_model)

            m_gradient = 1/n_rows * np.dot(X.T, (y_pred - y))
            b_gradient = 1/n_rows * np.sum(y_pred - y)

            self.m = self.m - self.lr * m_gradient 
            self.b = self.b - self.lr * b_gradient 
        

    def predict(self, X):
        LR_model = np.dot(X, self.m) + self.b
        y_pred = self._sigmoid(LR_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_class)

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))