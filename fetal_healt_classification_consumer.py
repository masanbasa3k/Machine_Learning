import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data
df = pd.read_csv('datasets/fetal_health_data.csv')

# set X an y
X = df.drop(["fetal_health"], axis=1)
y = df.fetal_health.values

# normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, stratify=y)

# MODELS
# prepare models
models = []
from sklearn.linear_model import LogisticRegression
models.append(('LR', LogisticRegression()))
from sklearn.ensemble import RandomForestClassifier
models.append(('RFC', RandomForestClassifier()))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
models.append(('LDA', LinearDiscriminantAnalysis()))
from sklearn.neighbors import KNeighborsClassifier
models.append(('KNN', KNeighborsClassifier()))
from sklearn.tree import DecisionTreeClassifier
models.append(('CART', DecisionTreeClassifier()))
from sklearn.naive_bayes import GaussianNB
models.append(('NB', GaussianNB()))
from sklearn.svm import SVC
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results.append(score)
    names.append(name)
    print(f'{name} : %{(score * 100):.2f}')

# graph
plt.barh(names, results, color='green')
plt.xlabel('Result (%)')
plt.ylabel('Name')
plt.title('Bar Plot Example')
plt.grid()

for index, value in enumerate(results):
    pvalue = f"%{(value * 100):.2f}"
    plt.text(value, index, pvalue)
plt.show()