import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scipy.optimize as opt
from sklearn.datasets import make_blobs

# generate 2d classification dataset
print("Generating data...")
X, y = make_blobs(n_samples=250, centers=3, n_features=2)

data = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]}).dropna().reset_index()

learning_rate = 0.001
iterations = 500

import gradient_descent
import square_root
import sklearn_gradient_descent
import tensorflow_example


gradient_descent.init_data(data)
square_root.init_data(data)
sklearn_gradient_descent.init_data(data)


print("Computing sqrt...")
square_root.compute()
print("Training gradient_descent...")
gradient_descent.train(data, iterations, learning_rate)
sklearn_gradient_descent.train()
tensorflow_example.train(data, iterations, learning_rate)


plt.scatter(data['x'], data['y'], label="Data")
gradient_descent.prep_plot()
square_root.prep_plot()
sklearn_gradient_descent.prep_plot()
tensorflow_example.prep_plot()
plt.title("Training results")
plt.legend()
plt.show()