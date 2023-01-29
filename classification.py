import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# data 
data = pd.read_csv('datasets/diabetes.csv')

# ön bi bakış atmana yarıyor null deger var mi yok mu bakıyorsun
#data.info(verbose=True)

# bu da count mean std min max değerlerini gösteriyo
#print(data.describe().T)

#null geğerlerini olup olmadığını gösteriyo
#print(data.isnull().sum())

data_copy = data.copy(deep=True)

data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.NaN)
#sıfır olan değerler datamızı çok etkileyecegi için düzeltmemiz lazım

#print(data_copy.isnull().sum())

data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)

#sns.pairplot(data_copy, hue='Outcome')
#plt.show()
# bunun ile bütün dataların ikili ilişkilerini görüyoruz buna bakarak cluster mı classification mı ne yapabileceğimize bakıyoruz

#kolerasyon için heatmap
#plt.figure(figsize=(12, 10))
#p = sns.heatmap(data_copy.corr(), annot=True, cmap='RdYlGn')
#plt.show()

# Standart Scaler
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

# geting x and y
X = pd.DataFrame(sc_x.fit_transform(data_copy.drop(['Outcome'], axis=1)),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = data_copy.Outcome

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
plt.show()
'''

# k = 13 en iyi sonuc
knn = KNeighborsClassifier(13)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print(f'Score with k = 13 : %{(score * 100):.2f}')

# conf. matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)
ct = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins_name=True)
print(ct)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)