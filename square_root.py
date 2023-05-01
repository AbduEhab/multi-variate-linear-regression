import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = 0
x_lin = 0
y_model = 0

def init_data(ext_data):
    global data
    data = ext_data

# square root regression
def func(x, A, c, d):
    return A * np.exp(c*x) + d

def compute():
    global x_lin, y_model

    x_lin = np.linspace(0, data['y'].max(), len(data['x']))

    p0 = [-1, -3e-3, 1]
    w, _ = opt.curve_fit(func, data['x'], data['y'], p0=p0)
    print("Estimated Parameters", w)

    # Model
    y_model = func(x_lin, *w) # same as ...w

def plot():
    plt.scatter(data['x'], data['y'], label="Data")
    plt.plot(x_lin, y_model, color='red', label="Fit")
    plt.xlim([0, data['x'].max()])
    plt.legend(['Model', 'Line of best fit'])
    plt.title("Least squares regression")
    plt.show()

def prep_plot():
    plt.plot(x_lin, y_model, color='green', label="Square root regression")
