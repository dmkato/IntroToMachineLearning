from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def get_data(type):
    with open('data/housing_{}.txt'.format(type), 'r') as data_file:
        data_strings = [l.split() for l in data_file.readlines()]
    return data_strings

def clean_data(data, dummy, r_count):
    data_floats = [[float(c) for c in l] for l in data]
    data_transposed = transpose(data_floats)
    if dummy:
        dummies = [1.0 for i in range(len(data_transposed[0]))]
        data_transposed = [dummies] + data_transposed
    if r_count > 0:
        for i in range(r_count):
            r_feature = [np.random.normal() for i in range(len(data_transposed[0]))]
            data_transposed = [r_feature] + data_transposed
    num_rows = len(data_transposed) - 1
    features = np.array(data_transposed[:num_rows])
    desired_outputs = np.array(data_transposed[num_rows])
    return (features, desired_outputs)

def get_clean_data(type, dummy, r_count):
    return clean_data(get_data(type), dummy, r_count)

def transpose(m):
    return [[j[i] for j in m] for i in range(len(m[0]))]

def optimal_weight_vector(x, y):
    # w = (X^T * X)^-1 * X^T * Y)
    X = np.array(x).transpose()
    Y = np.array(y).transpose()
    t1 = np.linalg.inv(np.matmul(X.transpose(), X))
    t2 = np.matmul(X.transpose(), Y)
    return np.matmul(t1, t2)

def apply_weight_vector(x, w):
    Yp = []
    for i, _ in enumerate(x[0]):
        median_p = 0
        for j, r in enumerate(x):
            median_p += r[i] * w[j]
        Yp += [median_p]
    return Yp

def Avg_Sq_Err(Y, Yp):
    SE = [(a - p) ** 2 for (a, p) in zip(Y, Yp)]
    SSE = reduce(lambda a, b: a+b, SE)
    ASE = SSE / len(Y)
    return ASE

def get_ASEs(dummy=True, r_count=0):
    print("d = {}".format(r_count))
    X, Y = get_clean_data('train', dummy, r_count)
    w = optimal_weight_vector(X, Y)
    # print("    Optimal Weight Vector: {}".format(w))

    # Apply weight vector to training data
    Yp = apply_weight_vector(X, w)
    ASE = Avg_Sq_Err(Y, Yp)
    print("    Training Data ASE: {}".format(ASE))

    # Apply weight vector to test data
    tX, tY = get_clean_data('test', dummy, r_count)
    tYp = apply_weight_vector(tX, w)
    tASE = Avg_Sq_Err(tY, tYp)
    print("    Test Data ASE: {}".format(tASE))
    return (ASE, tASE)

def plot_ASEs(ASEs):
    train_ASEs = [i[0] for i in ASEs]
    test_ASEs = [i[1] for i in ASEs]
    x_ax = range(len(ASEs))
    plt.plot(x_ax, train_ASEs, 'r', label='train')
    plt.plot(x_ax, test_ASEs, 'b', label='test')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    get_ASEs()
    get_ASEs(dummy=False)
    ASEs = [get_ASEs(r_count=i) for i in range(60)]
    plot_ASEs(ASEs)
