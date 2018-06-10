## HAC

1. The goal of clustering is to organize the data into subgroups such that all samples in a sub group are very similar to each other and very different from the examples in other sub groups.
    Some examples of clustering are product or movie suggestions, website categorization,

2. Single Link: Closest points in the clusters
   Complete Link: Furthest points in the clusters
   Average Link: Average distance between cluster pairs

3. Inverse of above

4. Single Link

                     ----------
                    |          |
               -----------     |
              |           |    |
           -------        |    |
          |       |       |    |
        ----     ---      |    --
       |    |   |   |     |   |  |
       0  0.45  1  1.2  2.55  4  5  

   Complete Link


               -------------
              |             |
           -------         ----
          |       |       |    |
        ----     ---      |    --
       |    |   |   |     |   |  |
       0  0.45  1  1.2  2.55  4  5  

5. The single link dendrogram is deeper than the complete ink dendrogram by 2 and the complete link is more balanced than the single link.


## K-Means

6. Because of the monotocity property, each iteration of K-mean strictly decreases the SSE until convergence. This is because at some point the new mean will be so close to the previous mean that none of the data points will be reclassified.

7.

   C1 = 4; C2 = 5
   |                        || |          
    0  0.45  1  1.2  2.55  4  5

   C1 = 2; C2 = 5
   |                     ||    |          
    0  0.45  1  1.2  2.55  4  5    

   C1 = 1.27; C2 = 4.5
   |                     ||    |          
    0  0.45  1  1.2  2.55  4  5    

8. No, the final partition is heavily dependent on the initial cluster center.

9. The k-medoid algorithm chooses the datapoint that is closest to the center, not the computed center of the data. This increases computation time as we need to search fo the datapoint in the center, but provides a more robust model less sensitive to outliers.

10. SSE always decreases as k increases because each datapoint will be closer to it's clusters mean.

11. If the initial seeds are bad, the final clusters will not be close to the global optimum, leading to a higher misclassification rate. We can avoid them by choosing seeds that are far apart and to test a series of random initializations to find the one with the smallest SSE.

12. Yes, with the correctly chosen seed points, in each scenario, we could get objective values that are not accurate.

## Gaussian Mixture Modeling

13. If a GMMm covariance matrix is restricted to diagonal, we can only get circular and elliptical cluster shapes, not tilted ellipticals.

15. Expectation Maximization's E-step corresponds to the K-means' reassignment and the M-step corresponds to the re-centering. K-means is Expectation Maximization with probabilities of 100%.

## PCA

16. PCA's objective is to reduce the dimensions of a dataset while retaining as much information possible. This reduction is done independent of class labels, and thus is an unsupervised learning algorithm.

17. Fit a line to the data such that the line has the most variance. Then find the line orthogonal to that line and that is your new y-axis.

18. The nth eigenvalue is the variance of after projecting the dat to the nth eigenvector.

19. S equals the sum of the first k eigenvalues divided by the sum of the first d eigenvalues where d if the number of the dimensions of the dataset. Choose the smallest k such that S is greater than the threshold aka 90%.

20. Because the PCA values are no longer indicative of a feature, it can be hard to draw class separations on these derived features.

## Evaluation of Clustering

21. Rand index: looks at pairs of instances and whether they are paired together or not.
Adjusted Rand index: Rand index contrasted against expected rand index value achieved by a random partition
Normalized Mutual Information: used in decision tree; 0 = cluster is meaningless; 1 = cluster holds information
Purity: accuracy when each datapoint is given the class label of its cluster

22. Rand index: 0 - 1
Adjusted Rand index: 0 - 1
Normalized Mutual Information: 0 -1
Purity: 0.5 - 1

23. Rand index: Very sensitive to c'
Adjusted Rand index: Less sensitive to c'
Normalized Mutual Information: Not very sensitive to c'
Purity: sensitive to c'

## Anomaly detection

24. Accuracy measure can be misleading because with anomalies, a model can achieve 99.9% if it ignores them.

25. F1 Measure is the average of the precision and the recall of a prediction. Precision is the number of true positives divided by number of true and false positives predicted. Recall is the number of true positives divided by the number of actual positives in the ground truth.

26. 
