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

def knn(train_set, test, k):
    ns = find_neighbors(train_set, test, k)
    d = {-1: 0, 1: 0}
    for n in ns:
        d[n] += 1
    return max(d.items(), key=operator.itemgetter(1))[0]

def accuracy(test_set, preds):
    c = 0
    for t, p in zip(test_set, preds):
        if t.y != p:
            c += 1
    return (len(test_set) - c) / len(test_set) * 100

if __name__ =='__main__':
    preds = []
    k = get_k()
    train_set = get_data('train')
    test_set = get_data('test')
    preds = [knn(train_set, test, k) for test in test_set]
    a = accuracy(test_set, preds)
    print(a)
