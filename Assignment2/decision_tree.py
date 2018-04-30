# By Daniel Kato and Nathan Shepherd
#
# Instructions:
#  1. Place data files directly in folder with python files
#  2. Source the python3.5 venv by running `source /scratch/cs434spring2018/env_3.5/bin/activate`
#  3. Run decision_tree with command `python3 decision_tree.py`

import sys
import operator
import numpy as np
import math
from knn import Data
# import matplotlib.pyplot as plt

class Node:
    def __init__(self, data, depth, feature=None, theta=None):
        self.data = data
        self.depth = depth
        self.feature = feature
        self.theta = theta
        self.d_class = self.majority_class(data)
        self.l = None
        self.r = None

    def majority_class(self, data):
        if self.theta != None:
            return None
        pos = [d.y for d in data if d.y == 1]
        neg = [d.y for d in data if d.y == -1]
        return max((1, len(pos)), (-1, len(neg)), key=lambda i: i[1])[0]

    def print_tree(self):
        print('Feature: {}, Theta: {}, Depth: {}, Class: {}'.format(self.feature, self.theta, self.depth, self.d_class))
        print("data: {}".format([d.y for d in self.data]))
        if self.l: self.l.print_tree()
        if self.r: self.r.print_tree()

def get_data(type):
    with open('./knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
    return data_set

def entropy(data_set):
    neg_count = [d for d in data_set if d.y == -1]
    p_neg = len(neg_count) / float(len(data_set))
    p_pos = (len(data_set)-len(neg_count)) / float(len(data_set))
    t1 = -(p_neg * math.log(p_neg)) if p_neg else -p_neg
    t2 = -(p_pos * math.log(p_pos)) if p_pos else -p_pos
    return t1 + t2

def information_gain(s_data_set, feature, d_idx):
    pe = entropy(s_data_set)
    l, r = s_data_set[d_idx:], s_data_set[:d_idx+1]
    wl = len(l)/pe if pe > 0 else 0
    wr = len(r)/pe if pe > 0 else 0
    lt = entropy(l) * wl
    rt = entropy(r) * wr
    return pe - (lt + rt)

def optimal_split(data_set, f):
    s_data_set = sorted(data_set, key=lambda d: d.x[f])
    ig_max = (-sys.maxsize - 1, -sys.maxsize - 1)
    for i, _ in enumerate(data_set[:-1]):
        theta = (data_set[i].x[f] + data_set[i+1].x[f]) / 2
        ig = information_gain(s_data_set, f, i)
        if ig > ig_max[0]:
            ig_max = (ig, theta)
    return ig_max

def decision(data_set):
    # For all available features to split on
    feature_splits = []
    for feature in range(30):
        # Find optimal split
        opt_split = optimal_split(data_set, feature)
        feature_splits += [(feature, opt_split[0], opt_split[1])]
    return max(feature_splits, key=lambda fs: fs[1])

def all_same_class(dataset):
    pos = [d for d in dataset if d.y == -1]
    if len(pos) == len(dataset) or len(pos) == 0:
        return True

def split(data_set, feature, theta):
    l = [d for d in data_set if d.x[feature] < theta]
    r = [d for d in data_set if d.x[feature] >= theta]
    return l, r

def build_tree(dataset, max_depth, depth=0, d_class=1):
    if all_same_class(dataset) or depth == max_depth:
        return Node(dataset, depth)
    feature, ig, theta = decision(dataset)
    n = Node(dataset, depth, feature, theta)
    l, r = split(dataset, feature, theta)
    n.l = build_tree(l, max_depth, depth+1)
    n.r = build_tree(r, max_depth, depth+1)
    return n

def error_ratio(subarr):
    pos_count = len([d for d in subarr if d.y ==1])
    neg_count = len(subarr) - pos_count
    m = min(pos_count, neg_count)
    return m / len(subarr)

def test_decision(feature, theta, data_set):
    l, r = split(data_set, feature, theta)
    return (error_ratio(l) + error_ratio(r)) * 100

def decision_stump():
    train_data = get_data('train')
    test_data = get_data('test')
    feature, ig, theta = decision(train_data)
    print("Selected Feature: {}".format(feature))
    print("Information Gain: {}".format(ig))
    print("Theta: {:.2f}".format(theta))
    error = test_decision(feature, theta, test_data)
    print("Percent Error: {:.2f}".format(error))

def traverse_tree(root, features):
    if root.d_class != None:
        return root.d_class, root.depth

    if features[root.feature] < root.theta:
        return traverse_tree(root.l, features)
    else:
        return traverse_tree(root.r, features)

def test_tree(root, data):
    misclassified = 0
    for d in data:
        result, depth = traverse_tree(root, d.x)
        if result != d.y:
            misclassified += 1
    return misclassified

def decision_tree(max_depth, train_data, test_data):
    root = build_tree(train_data, max_depth)
    train_err = test_tree(root, train_data)
    test_err = test_tree(root, test_data)
    root.print_tree()
    print("Max Depth = {}".format(max_depth))
    print("Train Error: {}, Train Error: {}".format(train_err, test_err))
    return train_err, test_err

# def plot(results):
#     train_err = [d[0] for d in results]
#     test_err = [d[1] for d in results]
#     x_ax = range(1, len(results) + 1)
#     plt.plot(x_ax, train_err, 'r', label='Train Error')
#     plt.plot(x_ax, test_err, 'b', label='Test Error')
#     plt.xlabel("Depth")
#     plt.ylabel("Number of Errors")
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    train_data = get_data('train')
    test_data = get_data('test')
    results = [decision_tree(i, train_data, test_data) for i in range(1, 7)]
    # print(results)
    # plot(results)
