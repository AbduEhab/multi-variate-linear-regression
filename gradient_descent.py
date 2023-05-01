import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

data = 0
w = 0
b = 0
entry_points = 0

def init_data(ext_data):
    global data
    data = ext_data

    entry_points = len(data.iloc[:, 0].values.reshape(-1, 1))

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
    plt.title('Gradient descent regression')
    plt.legend(['Model', 'Line of best fit'])
    plt.show()

def prep_plot():
    plt.plot(data['x'], predict(data['x']), color='red', label="gradient descent")