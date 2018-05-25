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
    cv_matrix = np.cov(data.T)
    e_values, e_vectors = la.eig(cv_matrix)
    return e_values, e_vectors.T, mean

def show_eignvectors(top_features):
    """
    Calculate and show eignvectors for top_features
    Requires a call to plt.show()
    """
    fig = plt.figure(figsize=(8, 8))
    columns = len(top_features)
    for i, vect in enumerate(top_features):
        viewable_e_vect = np.array([i.real for i in vect])
        img = viewable_e_vect.reshape(28, 28)
        fig.add_subplot(1, columns, i+1)
        plt.imshow(img, cmap=plt.cm.Greys)

def show_sig_images(imgs):
    """
    Calculate and show eignvectors for top_features
    Requires a call to plt.show()
    """
    fig = plt.figure(figsize=(8, 8))
    columns = len(imgs)
    for i, raw_img in enumerate(imgs):
        img = raw_img.reshape(28, 28)
        fig.add_subplot(1, columns, i+1)
        plt.imshow(img, cmap=plt.cm.Greys)

def dimensionality_reduction(data, k):
    e_values, e_vectors, mean = pca(data)
    return e_vectors[:k]

def reduce_dimensions(data, e_vectors):
    return [[x.dot(v) for v in e_vectors] for x in data]

def get_most_significant_examples(transformed_data, data):
    idxs = []
    for i in range(len(e_vectors)):
        feature_i = [t[i] for t in transformed_data]
        idxs += [np.argmax(feature_i)]
    imgs = [data[i] for i in idxs]
    return imgs

if __name__ == "__main__":
    data = get_data("./unsupervised.txt")
    e_vectors = dimensionality_reduction(data, 10)
    show_eignvectors(e_vectors)
    transformed_data = reduce_dimensions(data, e_vectors)
    significant_imgs = get_most_significant_examples(transformed_data, data)
    show_sig_images(significant_imgs)
    plt.show()
