import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic import logisticRegression


cancer = datasets.load_breast_cancer()
X,y = cancer.data,cancer.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


classifier = logisticRegression(alpha=0.001,epoch=10000)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)

print("LR classification accuracy:",accuracy(y_test,predictions))
plt.scatter(X_test[:,0],X_test[:,1],c=predictions,cmap='winter',edgecolors='k',s=20)
plt.show()
 

