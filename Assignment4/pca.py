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

def pca(data):
    """
    Returns the eign vectors with the variances higher than threshold
    """
    mean = data.sum(axis=0) / data.shape[0]
    sum_arr = [np.outer(x - mean, x - mean) for x in data]
    cv_matrix = np.sum(sum_arr, axis=0) / data.shape[0]
    e_values, e_vectors = la.eig(cv_matrix)
    return e_values, e_vectors, mean

def show_eignvectors(top_features):
    """
    Calculate k means for the dataset and show the resulting image of each mean.
    Requires a call to plt.show()
    """
    fig = plt.figure(figsize=(8, 8))
    columns = len(top_features)
    for i, (val, vect) in enumerate(top_features):
        viewable_e_vect = np.array([i.real for i in vect])
        img = viewable_e_vect.reshape(28, 28)
        fig.add_subplot(1, columns, i+1)
        plt.imshow(img, cmap=plt.cm.Greys)
    plt.show()

def dimensionality_reduction(data, k):
    e_values, e_vectors, mean = pca(data)
    arr = list(zip(e_values, e_vectors))
    sorted_arr = sorted(arr, key=lambda i:i[0], reverse=True)
    top_features = sorted_arr[:k]
    for val, vect in top_features:
        print(val)
    show_eignvectors(top_features)

if __name__ == "__main__":
    data = get_data("./unsupervised.txt")
    dimensionality_reduction(data, 10)
