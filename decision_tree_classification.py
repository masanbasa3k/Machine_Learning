import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data 
data = pd.read_csv('datasets/heart_classification_data.csv')

Sick = data[data.output == 1]
NSick = data[data.output == 0]

x_data = data.drop(["output"], axis=1)
y = data.output.values

#normalization
X = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

# model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# train
dt.fit(X_train, y_train)

# predict
y_pred = dt.predict(X_test)

# score 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#dogruluk matrixini veriyo 

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
