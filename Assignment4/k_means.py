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

def show_image(x):
    """
    Displays the first image in the data set
    USES MATPLOTLIB
    """
    plt.imshow(x.reshape(28, 28), cmap=plt.cm.Greys)
    plt.show()

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

def get_k_random_datapoints(data, k):
    """
    Returns k random datapoints
    """
    idxs = np.random.choice(len(data), k)
    return [data[i] for i in idxs]

def get_SSE(clusters, means):
    SSE = 0
    for m, c in zip(means, clusters):
        ssum = sum([la.norm(x - m) ** 2 for x in c])
        SSE += ssum
    return SSE

def k_means(data, k):
    """
    Returns k disjoint clusters and their means
    """
    means = get_k_random_datapoints(data, k)
    p_means = [np.zeros(len(means[0])) for m in means]
    SSEs = []
    while diffs_above_threshold(means, p_means, threshold=0.0001):
        clusters = [[m] for m in means]
        for x in data:
            add_x_to_nearest_cluster(x, clusters, means)
        p_means = means
        means = update_means(clusters)
        SSEs += [get_SSE(clusters, means)]
    return means, SSEs

def show_means(data, k):
    """
    Calculate k means for the dataset and show the resulting image of each mean.
    Requires a call to plt.show()
    """
    means, SSE = k_means(data, k=k)
    fig = plt.figure(figsize=(8, 8))
    columns = len(means)
    for i, mean in enumerate(means):
        img = mean.reshape(28, 28)
        fig.add_subplot(1, columns, i+1)
        plt.imshow(img, cmap=plt.cm.Greys)

def plot_SSE(data, k):
    """
    Calculates k means for the data and plots the SSE of the training sessions
    """
    print("Plotting SSE")
    means, SSEs = k_means(data, k=k)
    x_ax = list(range(len(SSEs)))
    plt.figure(figsize=(5,3))
    plt.plot(x_ax, SSEs, 'r')
    plt.title("SSE")
    plt.xlabel("Iteration")
    plt.ylabel("SSE")

def plot_model_selection(SSEs, k_max):
    """
    Plots each SSE against each value of k
    """
    ks = list(range(2, k_max+1))
    plt.figure(figsize=(5,3))
    plt.plot(ks, SSEs, 'r')
    plt.title("Model Selection")
    plt.xlabel("k")
    plt.ylabel("SSE")

def model_selection(data, k_max):
    """
    Runs k means for each k ranging from 2 to k_max and plots the results
    """
    results = []
    for i in range(2, k_max+1):
        print(i)
        k_results = [k_means(data, k=i) for c in range(10)]
        SSEs = [r[1][0] for r in k_results]
        results += [min(SSEs)]
    plot_model_selection(results, k_max)

if __name__ == "__main__":
    data = get_data("./unsupervised.txt")
    plot_SSE(data, k=2)
    model_selection(data, k_max=10)
    show_means(data, k=10)
    plt.show()
