import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading
from tqdm import tqdm

# read data from csv file

data = pd.read_csv("data.csv").dropna().reset_index()

## split data into training and testing sets withouth using sklearn

# shuffle data

data = data.sample(frac=1).reset_index(drop=True)

# split data into training and testing sets

train_data = data.iloc[:int(0.8 * len(data))]

test_data = data.iloc[int(0.8 * len(data)):]

# gradient descent

# initialize parameters

theta = np.random.randn(1, 1)

# initialize hyperparameters

alpha = 0.01

iterations = 10000

# define gradient descent function

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in tqdm(range(iterations)):
        y_pred = X.dot(theta)
        theta = theta - (1 / m) * alpha * (X.T.dot((y_pred - y)))
    return theta

# define function to train model

def train_model(X, y, alpha, iterations):

    global theta 

    theta= gradient_descent(X, y, theta, alpha, iterations)

    return theta

# train model

X = train_data['x'].values.reshape(-1, 1)
y = train_data['y'].values.reshape(-1, 1)


theta = train_model(X, y, alpha, iterations)


# plot model

def plot_model():
    plt.figure(figsize=(8,6))
    plt.scatter(x=train_data['x'], y=train_data['y'], cmap='Set1', s=50)
    plt.plot(train_data['x'], theta[0]*train_data['x'], 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model')
    plt.show()

plot_model()

# predict values

X_test = test_data['x'].values.reshape(-1,1)
y_test = test_data['y'].values.reshape(-1,1)

y_pred = X_test.dot(theta)

# plot predictions

def plot_predictions():
    plt.figure(figsize=(8,6))
    plt.scatter(x=test_data['x'], y=test_data['y'], cmap='Set1', s=50)
    plt.scatter(x=test_data['x'], y=y_pred, cmap='Set1', s=50)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predictions')
    plt.show()

plot_predictions()



# if "name" == "__main__":

#     mthread = threading.Thread(target=plot_model)
#     mthread.start()

#     mthread2 = threading.Thread(target=plot_predictions)
#     mthread2.start()

#     mthread.join()
#     mthread2.join()

