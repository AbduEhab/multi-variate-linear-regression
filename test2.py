import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# read data from csv file

data = pd.read_csv("data.csv").dropna().reset_index()

entry_points = len(data.iloc[:, 0].values.reshape(-1, 1))

learning_rate = 0.01

iterations = 100


print(entry_points)

# shuffle data

# data = data.sample(frac=1).reset_index(drop=True)

w = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))
b = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))

def get_predictions(x, i):
    return w[i] * x + b[i]

def gradient_descent(x, y, data, learning_rate):
    w_gradient = 0
    b_gradient = 0

    for i in range(entry_points):
        y_hat = get_predictions(x, i)

        w_gradient += (-2 / entry_points) * x * (y - y_hat)
        b_gradient += (-2 / entry_points) * (y - y_hat)

    global w, b

    w = w - learning_rate * w_gradient
    b = b - learning_rate * b_gradient

def train(data, iterations, learning_rate):
    for i in tqdm(range(iterations)):
        gradient_descent(data['x'], data['y'], data, learning_rate)

def predict(x):
    return w * x + b

def plot():
    plt.scatter(data['x'], data['y'])
    plt.plot(data['x'], predict(data['x']), color='red')
    plt.show()

train(data, iterations, learning_rate)
plot()

