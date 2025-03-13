# Breast cancer (simple logistic regression)
import numpy as np
class logisticRegression():
    def __init__(self,alpha=0.001,epoch=10000):
        self.alpha = alpha
        self.epoch = epoch
        self.theta = None # theta == bias

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.theta = 0   
        for _ in range(self.epoch):
            linear_model = np.dot(X,self.weights) + self.theta
            y_predicted = self._sigmoid(linear_model)
            dw = (1/n_samples)*np.dot(X.T,(y_predicted-y))
            db = (1/n_samples)*np.sum(y_predicted-y)
            self.weights -= self.alpha*dw
            self.theta -= self.alpha*db
    def predict(self,X):
        linear_model = np.dot(X,self.weights) + self.theta
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))             
