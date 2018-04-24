import sys
import operator
import numpy as np
from Data import Data

def get_k():
    if len(sys.argv) == 3 and sys.argv[1] == '-k' and sys.argv[2].isdigit():
        return int(sys.argv[2])
    print("Usage: python3.5 knn.py -k <k value>")
    exit()

def get_data(type):
    with open('knn_data/knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
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

if __name__ =='__main__':
    k = get_k()
    train_set = get_data('train')
    test_set = get_data('test')

    # Training Error
    preds = [knn(train_set, test, k) for test in train_set]
    train_err = testing_error(train_set, preds)
    print('Training Error: {}/{}'.format(train_err, len(preds)))

    # Leave-One-Out Cross-validation-error
    preds = []
    for idx, test in enumerate(train_set):
        loo_set = train_set[:idx] + train_set[idx+1:]
        preds += [knn(loo_set, test, k)]
    train_err = testing_error(train_set, preds)
    print('LOOCVE Error: {}/{}'.format(train_err, len(preds)))

    # Testing Error
    preds = [knn(train_set, test, k) for test in test_set]
    test_err = testing_error(test_set, preds)
    print('Testing Error: {}/{}'.format(test_err, len(preds)))
