---
title: "CS 434: Implementation Assignment 3"
author: Daniel Kato & Nathan Shepherd
geometry: margin=1in
fontsize: 11pt
---

# Part 1
<!-- Daniel -->
<!-- What is a good learning rate that works for this data and this network structure?
Present your plots for different choices of learning rates to help justify your final choice of the learning rate.
How do you decide when to stop training? Evaluate your final trained network on the testing data and report its accuracy. -->
The learning rate that worked best with the sigmoid activation function was $lr = 0.1$. This is because as the loss approaches zero with $lr = 0.1$, the accuracy approaches 45, where as the loss approaches zero with $lr = 0.01$, the accuracy approaches 40. We decided to use a set number of epochs due to the amount of time needed to train the model using a validation set with ~5000 examples. The number we set was 1000 because this was high enough to see the loss flatten out, meaning we are nearing the lower limit of the function, and 1000 epochs took ~2 hours to complete. The testing accuracy of the model with $lr = 0.1$ was [insert test accuracy].

![Validation Accuracy Sigmoid ($lr = 0.1$)](./img/validation-accuracy-sigmoid-learning-rate-0.1.png)
![Validation Loss Sigmoid ($lr = 0.1$)](./img/validation-loss-sigmoid-learning-rate-0.1.png)

![Validation Accuracy Sigmoid ($lr = 0.01$)](./img/validation-accuracy-sigmoid-learning-rate-0.01.png)
![Validation Loss Sigmoid ($lr = 0.01$)](./img/validation-loss-sigmoid-learning-rate-0.01.png)

# Part 2
<!-- Nathan -->
<!-- What is a good learning rate that works for this data and this network structure?
Present your plots for different choices of learning rates to help justify your final choice of the learning rate.
How do you decide when to stop training? Evaluate your final trained network on the testing data and report its accuracy. -->


# Part 3
<!-- Please describe what you have tried for each of these parameters.
How do the choices influence the behavior of learning?
Does it change the convergence behavior of training?
How do they influence the testing performance?
Please provide a summary of the results and discuss the impact these parameters.
Note that your discussion/conclusion should be supported by experimental evidences like test accuracy, training loss curve, validation error curves etc -->


# Part 4
<!-- What do you observe in terms of training convergence behavior?
Do you find one structure to be easier to train than the other?
How about the final performance, which one gives you better testing performance?
Provide a discussion of the results.
Please provide necessary plots and figures to support your discussion. -->
