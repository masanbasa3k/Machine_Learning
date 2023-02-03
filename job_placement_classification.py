import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# data 
df = pd.read_csv('datasets/Job_Placement_Data.csv')

#df.info(verbose=True)

#print(df.describe().T)



#null check
#print((df.isnull().sum()/len(df)))

'''correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')
plt.show()'''

# Chacing values 
df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})
df["ssc_board"] = df["ssc_board"].map({"Central": 1, "Others": 0})
df["hsc_board"] = df["hsc_board"].map({"Central": 1, "Others": 0})
df["undergrad_degree"] = df["undergrad_degree"].map({"Sci&Tech": 1, "Comm&Mgmt": 0, "Others": 2})
df["hsc_subject"] = df["hsc_subject"].map({"Commerce": 1, "Science": 0, "Arts": 2})
df["work_experience"] = df["work_experience"].map({"Yes": 1, "No": 0})
df["specialisation"] = df["specialisation"].map({"Mkt&HR": 1, "Mkt&Fin": 0})

df = df.drop(["gender"], axis=1)

X = df.drop(["status"], axis=1)
y = df.status.values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0, stratify=y)
# stratify daha dengeli bir seçim yapabilmek için

# model
from sklearn.neighbors import KNeighborsClassifier
'''
# en iyi k yi bulmak icin bunu yapiyoruz

test_score = []
train_score = []

for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))

# max scores
max_train_score = max(train_score)
train_score_ind = [i for i, v in enumerate(train_score) if v == max_train_score]
 
max_test_score = max(test_score)
test_score_ind = [i for i, v in enumerate(test_score) if v == max_test_score]


print(f'Max Train Score : %{max_train_score * 100} - K : {list(map(lambda x: x+1, train_score_ind))}')
print(f'Max Test Score : %{max_test_score * 100} - K : {list(map(lambda x: x+1, test_score_ind))}')

# graph

plt.figure(figsize=(12, 5))
p = sns.lineplot(train_score, marker="*", label="Train Score")
p = sns.lineplot(test_score, marker="o", label="Test Score")
plt.show()'''


# k = 9 en iyi sonuc
knn = KNeighborsClassifier(9)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(f'Score with k = 9 : %{(score * 100):.2f}')

# conf. matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)
ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins_name=True)
print(f"confusion matrix : {ct}")

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(f"classification report : {cr}")


features1 = np.array([[67.00,	1, 91.00, 1, 1,	58.00, 1, 1, 55.0, 1, 58.80]])# exepting 1
features2 = np.array([[21.00,	0, 43.00, 0, 0,	12.00, 0, 0, 23.0, 0, 34.80]])# exepting 0
features3 = np.array([[50.00,	0, 35.00, 1, 0,	95.00, 0, 0, 30.0, 1, 45.80]])# who knows
prediction1 = knn.predict(features1)
prediction2 = knn.predict(features2)
prediction3 = knn.predict(features3)
print(f"Prediction 1 : {prediction1}, Prediction 2 : {prediction2}, Prediction 3 : {prediction3}")