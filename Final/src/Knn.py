# By Daniel Kato and Nathan Shepherd
#
# Instructions:
#  1. Place data files directly in folder with python files
#  2. Source the python3.5 venv by running `source /scratch/cs434spring2018/env_3.5/bin/activate`
#  3. Run knn with command `python3 knn.py`
import sys
import operator
import numpy as np

class Knn:
    def __init__(self):
        self.optimal_k = None
        self.train_data = None

    def train(self, train_data):
        self.norm_vect = get_norm_vect(train_data)
        self.train_set = normalize(train_data, self.norm_vect)
        self.optimal_k = 7 # model_selection(self.train_set)
        print('Optimal K:', self.optimal_k)

    def test(self, test_data):
        test_set = normalize(test_data, self.norm_vect)
        preds = [knn(self.train_set, test[:-1], self.optimal_k) for test in test_set]
        test_err = testing_error(test_set, preds)
        print('Test Accuracy:', (len(test_set) - test_err) / len(test_set))
        return preds


def get_data(type):
    with open('./knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    return data_set

def get_norm_vect(data):
    # Returns array of (feature_min, feature_max)
    norm_vect = [[i, i] for i in data[0][:-1]]
    for row in data:
        for feature, norm in zip(row[:-1], norm_vect):
            norm[0] = min(norm[0], feature)
            norm[1] = max(norm[1], feature)
    return norm_vect

def normalize_row(features, norm_vect):
    return [(float(f) - n[0]) / (n[1] - n[0]) for f, n in zip(features, norm_vect)]

def normalize(data, norm_vect):
    data_set = [normalize_row(l[:-1], norm_vect) + [l[-1]] for l in data]
    return data_set

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def find_neighbors(train_set, test, k):
    dists = [(dist(test, train[:-1]), train[-1]) for train in train_set]
    ns = sorted(dists)[:k]
    return [n[1] for n in ns]

def nearest_neighbor(ns):
    d = {0: 0, 1: 0}
    for n in ns:
        d[int(n)] += 1
    pred = max(d.items(), key=operator.itemgetter(1))[0]
    certainty = d[pred] / len(ns)
    return certainty, pred

def knn(train_set, test, k):
    ns = find_neighbors(train_set, test, k)
    return nearest_neighbor(ns)

def batch_knn(data_set):
    preds = [knn(train_set, test, k) for test in train_set]
    train_err = testing_error(train_set, preds)
    print('Training Error: {}/{}'.format(train_err, len(preds)))

def testing_error(test_set, preds):
    c = 0
    for t, (s, p) in zip(test_set, preds):
        if t[-1] != p:
            c += 1
    return c

def test_knn_with_k(train_set, k):
    # Leave-One-Out Cross-validation-error
    preds = []
    for idx, test in enumerate(train_set):
        loo_set = train_set[:idx] + train_set[idx+1:]
        preds += [knn(loo_set, test[:-1], k)]
    return testing_error(train_set, preds), k

def model_selection(train_set):
    results = []
    l = len(train_set)
    for k in range(1, 30):
        print('k = {}'.format(k))
        results += [test_knn_with_k(train_set, k)]
        print('Accuracy:', (l - results[-1][0])/ l)
    best_k = min([(r, k) for r, k in results])
    return best_k[1]
