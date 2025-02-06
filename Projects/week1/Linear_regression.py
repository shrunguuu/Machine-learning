from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#load the dataset
df = pd.read_csv("BostonHousing.csv")

#features:
x = df.drop(columns=["medv"])  # 'MEDV' is the target column (median house price)
y = df["medv"]
print(x.shape)
print(y.shape)

#split the dataset (80-20)
X_train, X_test, Y_train , Y_test = train_test_split(x,y,test_size=0.2)

#algorithm
l_reg = linear_model.LinearRegression()
plt.scatter(x.iloc[:, 5], y)
plt.show()

#train
model = l_reg.fit(X_train,Y_train)
predictions = model.predict(X_test)
print(f"predictions:{predictions}")
print(f"R^2 value:{l_reg.score(x,y)}")
print(f"co-efficient:{l_reg.coef_}")
print(f"intercepts:{l_reg.intercept_}")
