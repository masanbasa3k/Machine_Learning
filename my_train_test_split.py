import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def my_train_test(X, y, test_size):# test_size should be between 1 and 100
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, test_size)
    
    X_train = X[split]
    X_test = X[~split]
    y_train = y[split]
    y_test = y[~split]
    
    return X_train, X_test, y_train, y_test

# importing data
data = pd.read_csv("datasets/linear_regression_data.csv")

experience = data['experience'].values.reshape(-1, 1)
salary = data['salary'].values.reshape(-1, 1)

# data split

# Sklearn
import sklearn.model_selection as ms
X_train1, X_test1, y_train1, y_test1 = ms.train_test_split(experience, salary, test_size=1/3, random_state=0)

# Mine
X_train, X_test, y_train, y_test = my_train_test(experience, salary, test_size=66)

print('X_train')
print(f"Sklearn : {X_train1}, Sklearn len : {len(X_train1)}")
print(f"Mine : {X_train}, Mine len : {len(X_train)}")


# graph
plt.scatter(X_train1, y_train1, color='r', alpha=0.5)
plt.scatter(X_train, y_train, color='b', alpha=0.5)
plt.show()