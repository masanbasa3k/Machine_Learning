import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data
df = pd.read_csv('datasets/credit_score_classification_data.csv')

# df.info(verbose=True)
# print(df.isnull().sum()) # its all zero

# convert data to int
df["Credit_Score"] = df["Credit_Score"].map({"Standard": 1, "Poor": 0, "Good": 2})

# extraction the useless dates
# usesless = "ID", "Customer_ID", "Month", "Name", "Age", "SSN", "Occupation", "Type_of_Loan", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly", "Payment_Behaviour"
X = df.drop(["Credit_Score", "ID", "Customer_ID", 
             "Month", "Name", "Age", "SSN", "Occupation", 
             "Type_of_Loan", "Changed_Credit_Limit", 
             "Num_Credit_Inquiries", "Credit_Utilization_Ratio", 
             "Payment_of_Min_Amount", "Total_EMI_per_month", 
             "Amount_invested_monthly", "Payment_Behaviour"], axis=1)
X["Credit_Mix"] = X["Credit_Mix"].map({"Standard": 1, "Bad": 0, "Good": 2})
y = df.Credit_Score.values

# normalization
X = (X - np.min(X)) / (np.max(X) - np.min(X))

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# MODELS
results = []
names = []

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)
results.append(rfc_score * 100)
names.append("RFC")

# GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_score = gnb.score(X_test, y_test)
results.append(gnb_score * 100)
names.append("GNB")

# KNeighborsClassifier
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
# k = 5 is best answer (you can find the code how can i find the best k ,in my other ml codes)
knn = KNeighborsClassifier(5)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
results.append(knn_score * 100)
names.append("KNN")

# graph
plt.barh(names, results)
 
for index, value in enumerate(results):
    plt.text(value, index,
             str(value))
 
plt.show()
