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

def grad(v):

    return None

# def loss(g, y):
#     return None
#
# def batch_train2(X, Y, w):
#     lam = 0.0001
#     for x, y in zip(X, Y):
#         t1 = loss(grad(w * x), y)
#         t2 = (lam * (np.linalg.norm(w) ** 2)) / 2
#     return t1 + t2

def batch_train(X, Y, w):
    eta = 10 ** -7
    nabla = np.zeros(X.shape[1])
    for x, y in zip(X, Y):
        y_hat = 1 / (1 + np.exp(-w * x))
        nabla = nabla + ((y_hat - y) * x)
    w = w - (eta * nabla)
    w = w + (10**-7)*sum(w**2)
    return nabla, w

def get_percent_correct(Y, R):
    c = 0
    for y, r in zip(Y, R):
        c = c + 1 if y == r else c
    return (c / Y.shape[0]) * 100

def test(w, type):
    X, Y = get_data(type)
    R = [0 if np.sum(x * w) < 0 else 1 for x in X]
    c = get_percent_correct(Y, R)
    return c

def training_loop():
    X, Y = get_data('train')
    w = np.zeros(X.shape[1])
    eps = 350
    d = eps + 1
    i = 0
    results = []
    print("{: <7} {: <15} {: <14} {: <10}".format("Batch", "Train Percent", "Test Percent", "Delta Norm"))
    while np.linalg.norm(d) > eps:
        i += 1
        d, w = batch_train(X, Y, w)
        results += [(test(w, "train"), test(w, "test"))]
        print("{: <7} {: <15.4f} {: <14.4f} {: <10.5f}".format(i, results[i-1][0], results[i-1][1], np.linalg.norm(d)))
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
