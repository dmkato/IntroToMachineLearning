from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

def get_data(type):
    with open('data/usps-4-9-{}.csv'.format(type), 'r') as data_file:
        data_strings = [l.split(',') for l in data_file.readlines()]
    data_ints = [[int(c) for c in l] for l in data_strings]
    targets = [i[256] for i in data_ints]
    return (np.array(data_ints), np.array(targets))

def sigmoid(w, x):
    return 1 / (1 + np.exp(-w * x))

def loss(y, y_hat, x):
    return ((y_hat - y) * x)

def reg(lam, w):
    return (lam * (sum(w) ** 2) / 2)

def batch_train(X, Y, w):
    eta = 10 ** -7
    lam = 10 ** 2
    delta = np.zeros(X.shape[1])
    for x, y in zip(X, Y):
        y_hat = sigmoid(w, x)
        delta_i = loss(y, y_hat, x) + reg(lam, w)
        delta = delta + delta_i
    w = w - (eta * delta)
    return delta, w

def get_percent_correct(Y, R):
    c = 0
    for y, r in zip(Y, R):
        c = c + 1 if y == r else c
    return (c / Y.shape[0]) * 100

def test(w, X, Y):
    R = [1 if np.sum(x * w) > 0.5 else 0 for x in X]
    c = get_percent_correct(Y, R)
    return c

def training_loop():
    X, Y = get_data('train')
    X_test, Y_test = get_data('test')
    w = np.zeros(X.shape[1])
    eps = 400
    d = eps + 1
    i = 0
    results = []
    print("{: <7} {: <15} {: <14} {: <10}".format("Batch", "Train Percent", "Test Percent", "Delta Norm"))
    while np.linalg.norm(d) > eps:
        i += 1
        d, w = batch_train(X, Y, w)
        results += [(test(w, X, Y), test(w, X_test, Y_test))]
        print("{: <7} {: <15.4f} {: <14.4f} {: <10.5f}".format(i, results[i-1][0], results[i-1][1], np.linalg.norm(d)))
    print(w)
    return results

def plot(r):
    train_results = [i[0] for i in r]
    test_results = [i[1] for i in r]
    x_ax = range(len(r))
    plt.plot(x_ax, train_results, 'r', label='train')
    plt.plot(x_ax, test_results, 'b', label='test')
    plt.xlabel("Batch Number")
    plt.ylabel("Percent Correct")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    results = training_loop()
    plot(results)
