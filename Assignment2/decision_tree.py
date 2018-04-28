import sys
import operator
import numpy as np
import math
from Data import Data
from Node import Node

def get_data(type):
    with open('knn_data/knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
    return data_set

def entropy(data_set):
    neg_count = [d for d in data_set if d.y == -1]
    p_neg = len(neg_count) / float(len(data_set))
    p_pos = (len(data_set)-len(neg_count)) / float(len(data_set))
    t1 = -(p_neg * math.log(p_neg)) if p_neg else -p_neg
    t2 = -(p_pos * math.log(p_pos)) if p_pos else -p_neg
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
    ig_max = 0
    for i, _ in enumerate(data_set[:-1]):
        theta = (data_set[i].x[f] + data_set[i+1].x[f]) / 2
        ig = information_gain(s_data_set, f, i)
        ig_max = max(ig, ig_max)
    return (ig_max, theta)

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

def build_tree(dataset, max_depth, depth=0):
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

def split(data_set, feature, theta):
    l = [d for d in data_set if d.x[feature] < theta]
    r = [d for d in data_set if d.x[feature] >= theta]
    return l, r

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

def decision_tree(max_depth):
    train_data = get_data('train')
    test_data = get_data('test')
    root = build_tree(train_data, max_depth)
    root.print_tree()

if __name__ == '__main__':
    decision_tree(4)
