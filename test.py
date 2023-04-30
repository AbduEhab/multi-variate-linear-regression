import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# read data from csv file

data = pd.read_csv("data.csv").dropna().reset_index()

entry_points = len(data.iloc[:, 0].values.reshape(-1, 1))

learning_rate = 0.01

iterations = 1000


print(entry_points)

# shuffle data

# data = data.sample(frac=1).reset_index(drop=True)

w = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))
b = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))

def calc_gradients(l_data, total_points, x, y, i):
    y_hat = get_predictions(x, i)

    return ((-2 / total_points) * x * (y - y_hat)), ((-2 / total_points) * (y - y_hat))

def compute_gradients(l_data, total_points):
    dw = 0
    db = 0

    for i in range(total_points):
        dwi, dbi = calc_gradients(l_data, total_points, l_data['x'][i], l_data['y'][i], i)
        dw += dwi
        db += dbi

    return dw, db

def update_parameters(l_data, total_points, learning_rate):
    dw, db = compute_gradients(l_data, total_points)

    global w, b

    w = w - learning_rate * dw
    b = b - learning_rate * db


def train(l_data, total_points, iterations, learning_rate):
    for i in tqdm(range(iterations)):
        update_parameters(l_data, total_points, learning_rate)

def predict(x):
    return w * x + b

def plot():
    plt.scatter(data['x'], data['y'])
    plt.plot(data['x'], predict(data['x']), color='red')
    plt.show()

train(data, entry_points, iterations, learning_rate)

print(w)

plot()
