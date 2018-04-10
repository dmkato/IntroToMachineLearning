from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def print_ex(x):
    ramp = ['$', '0', 'L', 'J', 'Y', 'z', 'v', 'n', 'r', 'f', '/', '|', ')', '{', '[', '?', '_', '~', '>', '!', 'I', ':', '^', "'", ' ']
    rx = [i//10 for i in x]
    for i in range(16):
        for j in range(16):
            print(ramp[rx[(16*i)+j] - 1], end='')
            print(ramp[rx[(16*i)+j] - 1], end='')
        print()

def get_data(type):
    with open('data/usps-4-9-{}.csv'.format(type), 'r') as data_file:
        data_strings = [l.split(',') for l in data_file.readlines()]
    data_ints = [[int(c) for c in l] for l in data_strings]
    targets = [1 if i[256] == 1 else -1 for i in data_ints]
    return (np.array(data_ints), np.array(targets))

def batch_train(X, Y, w):
    """
    X: Example Set
    Y: Training Solutions
    w: Weight Vector
    """
    n = 1
    d = np.zeros(X.shape[1])
    for x, y in zip(X, Y):
        u = w * x
        if np.sum(y * u) <= 0:
            d = np.subtract(d, x * y)
    d = np.divide(d, X.shape[1])
    w = np.subtract(w, n * d)
    return d, w

def train():
    X, Y = get_data('train')
    w = np.zeros(X.shape[1])
    eps = 4
    d = eps + 1
    i = 0
    while np.linalg.norm(d) > eps:
        i += 1
        print("Batch {}".format(i))
        d, w = batch_train(X, Y, w)
        print("delta norm = {}".format(np.linalg.norm(d)))
    return w

def get_percent_correct(Y, R):
    c = 0
    for y, r in zip(Y, R):
        c = c + 1 if y == r else c
    return (c / Y.shape[0]) * 100

def test(w):
    X, Y = get_data('test')
    R = [-1 if np.sum(x * w) < 0 else 1 for x in X]
    c = get_percent_correct(Y, R)
    print("Percent Correct {}".format(c))
    return w

if __name__ == "__main__":
    w = train()
    test(w)

## WHY THE FUCK IS IT STOPING AT BATCH 151!?
