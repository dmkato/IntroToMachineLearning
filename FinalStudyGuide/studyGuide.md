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

26. If we have a high cost for false positives and a low cost for false negatives, the favored ROC curve would be the one has a low false positive rate given a higher true positive rate. If we have a high cost for false negatives, we would want the curve that has the highest true positive rate.

27. Distance only takes distance into account, but density takes the density of the neighbors into account making the outlier decision dependent on the specific neighborhood the point is closest to.

## MDP

28. An MDP is composed of a finite set of states S, a finite set of actions A, a transition function T(s, a, s') = P(s'|s,a), and a reward function R(s).

29. The Markov property is the fact that actions do not depend on the history of the agent, only the current state. (Like a Markov chain)

30. A policy if a mapping of states to actions. It is what the MDP returns.

31. U(s), the utility state of s, measure the long term reward for being in state s, where the reward R(s) measure only the immediate reward of being in state s.

32. 1) An MDP agent for tic-tac-toe would be comprised of:
        State space: All possible configurations of the tic-tac-toe board. A board is comprised of 9 spots each which can be black, X, or O, giving us 3^9 states.
        Action space: All blank spaces in the current state can be acted upon.
        Transition Function: T(s, a, s') = P(s'|s,a)
                                         = 1 / (E-1)
        Reward: Any space adjacent to or inline with another space with the same tile.
    2) Because at each state, a player can put a tile on any of the 9 open spaces, and there are 3^9 states, the table would be 9 * 3^9 = 177,147
    3) Because the player is no longer acting randomly, the transition model is no longer uniform as the player will have a higher likelihood of placing a tile in certain positions at certain times.

33. a)
        U_0(1) = -1, U_0(2) = -2, U_0(3) = 0; // Rewards
        U_1(1) = U_0(1) + discount * max(prob2 * U_0(2) + probStay * U_0(1); prob3 * U_0(3) + probStay * U_0(1))

    b)

34. Passive learning: Agent has a fixed policy and is trying to learn utility of each state by observing the world.
    Active learning: Agent tries to find an optimal policy by acting in the world.

35. Exploration: Try some possibly random sequences of actions to improve knowledge of the environment; Kinda like training
    Exploitation: Execute an action that is learned to have a high payoff; Kinda like testing

36. Utility provides the max Q value from the state times the discount. Q values can provide us with model-less learning because we are only after the max Q values at each state

37. Adaptive Dynamic Programming uses O(|A||S|^2) while Q learning uses O(|A||S|)

## Ensemble

38. Unpruned decision trees have a lot of variance due to leaf nodes not being entirely accurate. This makes it a good candidate for bagging because bagging can reduce the variance of a model

39. Bootstrapping works by taking random, un-replaced samples over a normal distribution of data points from the dataset to compose a new dataset. Because each model is trained on a different dataset, the classifier will be different.

40. The only difference between a random forest and a bagged decision tree is in a random forest, only a random subset of features is considered at each node when making a split.

41. Bagging: Uses an ensemble of random samples from the dataset
             Reduces variance
             Robust against noise and outliers
    Boosting: Iteratively improved a model by weighting its mistakes (modify data distribution)
              Reduces bias and variance
              Can hurt performance with noise and outliers

42. If the model correctly classifies an example, it's weight is decreased by e^-a_t, and if it incorrectly classifies an example it's weight is increased by e^a_t.

43. ^

44. No, sampling reduces the impact of noise and outliers.

45. Yes, overtime outliers will receive very high weights, making them a priority and heavily skewing your models.

46. No, Bagging is safe and does hurt performance.

47. Yes, overfitting to outliers.
