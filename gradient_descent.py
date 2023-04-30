import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import make_blobs

# generate 2d classification dataset
X, y = make_blobs(n_samples=250, centers=10, n_features=2)

data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]}).dropna().reset_index()

entry_points = len(data.iloc[:, 0].values.reshape(-1, 1))

learning_rate = 0.001

iterations = 100

print("Entery points: ", entry_points)

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

w = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))
b = np.random.uniform(low=-0.01, high=0.01, size=(entry_points,))

def get_predictions(w, b, x):
    return w * x + b

def compute_gradients(local_w, local_b, l_data, total_points):
    gw = 0
    gb = 0

    for i in range(total_points):

        x = l_data.iloc[i].x
        y = l_data.iloc[i].y

        y_hat = get_predictions(local_w, local_b, x)

        gw += (-2 / total_points) * x * (y - y_hat)
        gb += (-2 / total_points) * (y - y_hat)

    return gw, gb

def update_parameters(l_data, total_points, learning_rate):
    global w, b

    gw, gb = compute_gradients(w, b, l_data, total_points)

    w = w - learning_rate * gw
    b = b - learning_rate * gb


def train(l_data, iterations, learning_rate):
    total_points = len(l_data)

    for i in tqdm(range(iterations)):
        update_parameters(l_data, total_points, learning_rate)

w.reshape(-1, 1)

def predict(x):
    return w * x + b

def plot():
    plt.scatter(data['x'], data['y'])
    plt.plot(data['x'], predict(data['x']), color='red')
    plt.title('Model & Line of best fit')
    plt.legend(['Model', 'Line of best fit'])
    plt.show()

train(data, iterations, learning_rate)

plot()