import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
data = pd.read_csv("datasets/linear_regression_data.csv")

experience = data['experience'].values.reshape(-1, 1)
salary = data['salary'].values.reshape(-1, 1)
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# hatasını alırsan pandas ile alakalı bir hatadır bunu şu şekilde çözebilirsin array.reshape(1, -1)

# importing algorithm
import sklearn.linear_model as lm
reg = lm.LinearRegression()

# data split
import sklearn.model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(experience, salary, test_size=1/3, random_state=0)

# train
reg.fit(X_train, y_train)

# predict

y_pred = reg.predict(X_test)

print(X_test)
print(y_pred)

# score
import sklearn.metrics as mt
score = mt.r2_score(y_test, y_pred)
print('Score :',score)

# graph
plt.scatter(experience, salary, color='r')
plt.scatter(X_test, y_pred, color='b')
plt.show()