import sys
import operator
import numpy as np
from Data import Data
from DTree import DTree

def get_data(type):
    with open('knn_data/knn_{}.csv'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
    return data_set

def entropy(data_set):


def information_gain():


def split(train_data):


def accuracy(test_data, preds):
    c = 0
    for t, p in zip(test_data, preds):
        if t.y != p:
            c += 1
    return (len(test_data) - c) / len(test_data) * 100

if __name__ == '__main__':
    train_data = get_data(train)
    test_data = get_data(test)
    decision = split(train_data)
    predictions = test_decision(decision, test_data)
    accuracy = accuracy(test_data, predictions)
