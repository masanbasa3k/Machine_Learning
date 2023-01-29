import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
data = pd.read_csv("datasets/random_forest_regression_data.csv")

X = data.iloc[:, 0].values.reshape(-1,1)
y = data.iloc[:, 1].values.reshape(-1,1)

# model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=40)

# fit
rf.fit(X, y)

# predict
x2 = np.arange( min(X), max(X), 0.01).reshape(-1,1)
y_pred = rf.predict(x2)

# graph
plt.scatter(X, y, color='r')
plt.plot(x2, y_pred, color='b')
plt.show()
