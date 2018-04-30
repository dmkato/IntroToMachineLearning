# By Daniel Kato and Nathan Shepherd
#
# Instructions:
#  1. Place data files directly in folder with python files
#  2. Source the python3.5 venv by running `source /scratch/cs434spring2018/env_3.5/bin/activate`
#  3. Run knn with command `python3 knn.py`

import sys
import operator
import numpy as np
from Data import Data
# import matplotlib.pyplot as plt

def get_k():
    if len(sys.argv) == 3 and sys.argv[1] == '-k' and sys.argv[2].isdigit():
        return int(sys.argv[2])
    print("Usage: python3.5 knn.py -k <k value>")
    exit()

def get_data(type):
    with open('./knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
    return data_set

def get_norm_vect():
    # Returns array of (feature_min, feature_max)
    data = get_data('train')
    norm_vect = [[i, i] for i in data[0].x]
    for row in data:
        for feature, norm in zip(row.x, norm_vect):
            norm[0] = min(norm[0], feature)
            norm[1] = max(norm[1], feature)
    return norm_vect

def normalize(features, norm_vect):
    return [(float(f) - n[0]) / (n[1] - n[0]) for f, n in zip(features, norm_vect)]

def get_norm_data(type, norm_vect):
    with open('./knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(normalize(l[1:], norm_vect), l[0]) for l in lines]
    return data_set

def dist(a, b):
    return np.linalg.norm(a - b)

def find_neighbors(train_set, test, k):
    dists = [(dist(test.x, train.x), train.y) for train in train_set]
    ns = sorted(dists)[:k]
    return [n[1] for n in ns]

def nearest_neighbor(ns):
    # TODO: POssibly weight based on distance
    d = {-1: 0, 1: 0}
    for n in ns:
        d[n] += 1
    return max(d.items(), key=operator.itemgetter(1))[0]

def knn(train_set, test, k):
    ns = find_neighbors(train_set, test, k)
    return nearest_neighbor(ns)

def batch_knn(data_set):
    preds = [knn(train_set, test, k) for test in train_set]
    train_err = testing_error(train_set, preds)
    print('Training Error: {}/{}'.format(train_err, len(preds)))

def testing_error(test_set, preds):
    c = 0
    for t, p in zip(test_set, preds):
        if t.y != p:
            c += 1
    return c

def knn_with_k(train_set, test_set, k):
    # Training Error
    preds = [knn(train_set, test, k) for test in train_set]
    train_err = testing_error(train_set, preds)
    print('Training Error: {}/{}'.format(train_err, len(preds)))

    # Leave-One-Out Cross-validation-error
    preds = []
    for idx, test in enumerate(train_set):
        loo_set = train_set[:idx] + train_set[idx+1:]
        preds += [knn(loo_set, test, k)]
    loocve_err = testing_error(train_set, preds)
    print('LOOCVE Error: {}/{}'.format(loocve_err, len(preds)))

    # Testing Error
    preds = [knn(train_set, test, k) for test in test_set]
    test_err = testing_error(test_set, preds)
    print('Testing Error: {}/{}'.format(test_err, len(preds)))
    return (train_err, loocve_err, test_err)

def model_selection(train_set, test_set):
    results = []
    norm_vect = get_norm_vect()
    train_set = get_norm_data('train', norm_vect)
    test_set = get_norm_data('test', norm_vect)
    for k in range(50):
        print('k = {}'.format(k))
        results += [knn_with_k(train_set, test_set, k)[1]]
        print()

    best_k = min([(r, i) for i, r in enumerate(results)])
    print("Optimal k: {}".format(best_k))

# def plot_knn(train_set, test_set):
#     all_results = [knn_with_k(train_set, test_set, k) for k in range(1, 51)]
#     train_err = [d[0] for d in all_results]
#     loocve_err = [d[1] for d in all_results]
#     test_err = [d[2] for d in all_results]
#
#     x_ax = range(len(all_results))
#     plt.plot(x_ax, train_err, 'r', label='Train Error')
#     plt.plot(x_ax, loocve_err, 'b', label='LOOCVE Error')
#     plt.plot(x_ax, test_err, 'g', label='Test Error')
#     plt.xlabel("k")
#     plt.ylabel("Number of Errors")
#     plt.legend()
#     plt.show()

if __name__ =='__main__':
    norm_vect = get_norm_vect()
    train_set = get_norm_data('train', norm_vect)
    test_set = get_norm_data('test', norm_vect)
    model_selection(train_set, test_set)
    # plot_knn()
