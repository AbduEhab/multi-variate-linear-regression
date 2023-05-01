import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

model = 0
data = 0
test = 0
y_hat = 0

def train(l_data, iterations, learning_rate):
    global model, y_hat, data, test

    data = l_data.sample(frac=1).reset_index(drop=True)
    # data = l_data

    data = data.drop(columns=['index'])

    # split data into test and train
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train = data.iloc[0:train_size]
    test = data.iloc[train_size:len(data)]

    # 1. Define the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units =64, input_dim=1, activation =tf.nn.relu ))
    model.add(tf.keras.layers.Dense(units =24, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units =16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units =1, activation=tf.nn.relu))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.MSE)

    # 3. Fit the model
    print("Training NN...")
    model.fit(x=train['x'], y=train['y'], batch_size=32, epochs=iterations, verbose=0)

    model.summary()

    # 4. Evaluate the model
    print("Evaluating NN...")
    error = model.evaluate(x=train['x'], y=train['y'], verbose=0)
    print("MSE: ", error)
    print("RMSE: ", np.sqrt(error))

    # 5. Make a prediction
    test = np.reshape(test, newshape=(-1, 1))
    y_hat = model.predict(test)

def prep_plot():
    global y_hat

    plt.scatter(test, y_hat, color='yellow', label="Tensorflow predictions")