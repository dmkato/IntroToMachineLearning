import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def get_data(filepath):
    """
    Returns data from filepath as an np.array of floats
    """
    with open(filepath, 'r') as f:
        lines = [l.strip().split(',') for l in f.readlines()]
    data_set = [np.array(l, dtype=float) for l in lines]
    return np.array(data_set)

def show_first_image(data):
    """
    Displays the first image in the data set
    USES MATPLOTLIB
    """
    print("done")
    plt.imshow(data[0].reshape(28, 28), cmap=plt.cm.Greys)
    plt.show()

def get_k_random_datapoints(data, k):
    """
    Returns k random datapoints
    """
    idxs = np.random.choice(len(data), k)
    return [data[i] for i in idxs]

def add_x_to_nearest_cluster(x, clusters, means):
    """
    Adds x to the cluster with the minimum distance to its mean
    """
    distances = [la.norm(x - m) for m in means]
    min_distance_idx = distances.index(min(distances))
    clusters[min_distance_idx] += [x]

def update_means(clusters):
    """
    Returns list of updated means cooresponding to the clusters
    """
    sums = [np.array(c).sum(axis=0) for c in clusters]
    means = [sum/len(c) for c, sum in zip(clusters, sums)]
    return means

def diffs_above_threshold(means, p_means, threshold):
    """
    Returns true if any of the differences between a cluster's mean and
        previous mean are above the threshold
    """
    diffs = [la.norm(p - m) for m, p in zip(means, p_means)]
    for diff in diffs:
        if diff > threshold:
            return True
    return False

def k_means(data, k):
    """
    Returns k disjoint clusters and their means
    """
    means = get_k_random_datapoints(data, k)
    clusters = [[m] for m in means]
    p_means = [np.zeros(len(means[0])) for m in means]
    while diffs_above_threshold(means, p_means, threshold=2):
        for x in data:
            add_x_to_nearest_cluster(x, clusters, means)
        p_means = means
        means = update_means(clusters)

if __name__ == "__main__":
    data = get_data("./unsupervised.txt")
    k_means(data, k=2)
