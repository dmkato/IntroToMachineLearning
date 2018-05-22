import matplotlib.pyplot as plt
import numpy

def get_data(type):
    with open('./unsupervised.txt'.format(type), 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [Data(l[1:], l[0]) for l in lines]
    return data_set

if __name__ == "__main__":
    
