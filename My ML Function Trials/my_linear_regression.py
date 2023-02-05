import numpy as np

class my_LinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        num = 0
        den = 0
        for xi, yi in zip(X, y):
            num += (xi - X_mean) * (yi - y_mean)
            den += (xi - X_mean) ** 2
        self.b_ = num / den
        self.a_ = y_mean - self.b_ * X_mean
        
    def predict(self, X):
        return self.a_ + self.b_ * X
    
    def score(self, y, predict):
        y_pred = predict
        y_mean = np.mean(y)
        u = ((y - y_pred)**2).sum()
        v = ((y - y_mean)**2).sum()
        return 1 - u/v

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from my_train_test_split import my_train_test_split

    # data  
    data = pd.read_csv("datasets/linear_regression_data.csv")
    experience = data['experience'].values.reshape(-1, 1)
    salary = data['salary'].values.reshape(-1, 1)   

    # train test sptlit
    X_train, X_test, y_train, y_test = my_train_test_split(experience, salary, train_size=66)

    # model
    lr = my_LinearRegression()
    lr.fit(X_train, y_train)

    # prediction
    y_pred = lr.predict(X_test)

    # score
    score = lr.score(y_test, y_pred)
    print(f'Score {(score * 100):.2f}:',)

    # graph
    plt.scatter(experience, salary, color='r')
    plt.scatter(X_test, y_pred, color='b')
    plt.show()