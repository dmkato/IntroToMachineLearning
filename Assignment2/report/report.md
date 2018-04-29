---
title: "CS 434: Implementation Assignment 2"
author: Daniel Kato & Nathan Shepherd
geometry: margin=1in
fontsize: 11pt
---

# KNN
1. We've implemented the KNN algorithm in Python, found in knn.py.

2.

# Decision Tree

1. Once again, we implemented the decision tree stump algorithm in Python, in decision_tree.py. Below is the representation of the stump found by our algorithm.  This shows that the optimal split was on feature 22 (x[21]), with $\theta=80.14$. This split divides the training data into the respective -1 or 1 categories, named as classes. The information gain is the difference of the entropies of each branch. The entropy of each branch is given by $-p_+\log_2 p_+-p_-\log_2 p_-$. The information of this particular split was calculated to be -238.976. Once split in this way, our training error percentage is 44.36% incorrect.  Our testing percentage is 43.66%.

    ![Tree of depth 1](./img/stump.png)
