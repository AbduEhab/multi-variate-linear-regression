import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = 0
y_model = 0
regressor = 0
X = 0
y = 0

def init_data(ext_data):
    global data
    data = ext_data

def train():
    global data, y_model, regressor, X, y

    data = data.drop(columns=['index'])

    X = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 1].values.reshape(-1, 1)

    regressor = LinearRegression().fit(X, y)

    print("Estimated Parameters", regressor.coef_, regressor.intercept_)

    # Model
    y_model = regressor.predict(X)

def plot(type):
    plt.scatter(data['x'], data['y'])
    plt.scatter(data['x'], y_model)
    plt.xlim([data['x'].min(), data['x'].max()])
    plt.title("sklearn LinearRegression regression")
    plt.legend(['Model', 'predictions'])
    plt.show()

def prep_plot():
    plt.scatter(data['x'], y_model, color='orange', label="sklearn LinearRegression regression")
    plt.xlim([data['x'].min(), data['x'].max()])