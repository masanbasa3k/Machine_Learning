import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
data = pd.read_csv("datasets/polinominal_regression_data.csv")

X = data['car_price'].values.reshape(-1,1)
y = data['car_max_speed'].values.reshape(-1,1)

# importing algorithm
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
reg = lm.LinearRegression()

polynominal_reg = PolynomialFeatures(degree=4)
x_polynominal = polynominal_reg.fit_transform(X, y)

#linear reg -> y = mx + b
# polynominal reg -> y = b0 + b1*x + b2*x^2 + b3*x^3 + ... + bn * x^n

# fit
reg.fit(x_polynominal, y)

# pred
y_pred = reg.predict(x_polynominal)

# score
from sklearn.metrics import r2_score
score = r2_score(y, y_pred)
print('score : ', score)

# graph
plt.plot(X, y_pred, color='b', label='poly')
plt.legend()
plt.scatter(X, y, color='r')
plt.xlabel('Arac Fiyat')
plt.ylabel('Arac Max Hiz')
plt.show()