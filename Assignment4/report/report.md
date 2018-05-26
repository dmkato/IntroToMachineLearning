---
title: "CS 434: Implementation Assignment 4"
author: Daniel Kato & Nathan Shepherd
geometry: margin=1in
fontsize: 11pt
---

# Part 1

Using the difference in means as our error, we converge on a threshold of 0.0001 for the rate of change of the SSE.
This can be seen by the chart below:

![SSE converging](img/kmeans-SSE-converge.png)

This convergence value is used for the rest of the results.
Running the k-means algorithm for values of $k=\{2,...,10\}$, ten times each, we record the data in the following chart.

![Model Selection](img/kmeans-modelselection.png)

Based off of this curve, we can see that 10 would be the correct k value.
This aligns with our expectations, because, if we were to classify them, the classes would be the ten digits.
Although our k-means algorithm works best on ten clusters, it doesn't seem to always find ten distinct numbers, as can be seen by the averages of the clusters.
These three sets represent a small sampling of the averages with 10 clusters.

![Cluster Averages 1](img/kmeans-clusteraverages1.png)

![Cluster Averages 2](img/kmeans-clusteraverages2.png)

![Cluster Averages 3](img/kmeans-clusteraverages3.png)


# Part 2
