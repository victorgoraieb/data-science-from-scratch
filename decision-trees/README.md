# Tree based methods - Study Notes

#### Learning Resources

- [Machine Learning University by Amazon](https://youtu.be/DtX1hN0FRfk)
- [Introduction to Statistical Learning]
- [Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.amazon.com.br/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)

### Introduction to Decision Trees

A decision tree is a **sequence of yes/no questions** about a input asked in a determined order.

The tree is composed by nodes: 
 - Split nodes (which have a question associated)
 - Leaf nodes (which have an answer associated)
 
For each question asked, we implement a horizontal/vertical decision boundary among the samples resulting in a non-linear decision boundary. Also, we need to measure how well we divided the samples through an impurity measurement

These type of models are greedy, that is, they don't look for the best overall solution but the best local one. In other terms, they pick a split in each node that maximizes the increase in purity

Also, we can define stopping criteria:
- maximum depth
- maximum number of leaves (regions that you assign a class)
- reached a minimum number of samples in each leaf
- once the leaf reaches a desired level of purity


### Impurity Functions

#### Impurity for classification

The impurity is a key concept for deciding whether or not to split a node. For evaluating the quality of a split, we look at the child nodes after a split node and want to the impurity to be as low as possible.

For classification problems, we usually opt for two main measures of impurity: the Entropy and the Gini. In terms of performance, they are extremely similar and lead to a small performance difference. In terms of results, we can multiply Gini by 2 and achieve a similar result as Entropy

$Entropy: impurity(p1, ..., pk) = - \sum \limits _{j=1} ^{k} p_{j} *log2(p_{j}) $

$Gini: impurity(p1, ..., pk) = \sum \limits _{j=1} ^{k} p_{j} *(1 - p_{j}) $

These functions can be visually represented with the x-axis being the pj probability (or the number of individuals of a certain class divided by the total) and the impurity itself. The key concept here is that both approaches have a maximum value at p1=0.5, that is, when we have equally distributed classes.

Since the logarithm is a more complex operation to compute, the Gini Index stands as the better option for day to day use.

For example, if we have two leaves and want to calculate the total impurity:
- Split 1: 55% of class 1 and 45% of class 2 (1000 samples)
- Split 2: 35% of class 1 and 65% of class 2 (300 samples)

- Using Entropy:
    - $impurity_{split_1} = -1 * ((550/1000) * log2(550/1000) + (450/1000) * log2(450/1000)) = 0.9927 $
    - $impurity_{split_2} = -1 * ((105/300) * log2(105/300) + (195/300) * log2(195/300)) = 0.9340 $
    - $impurity_{total} = 1000/1300 * 0.9927 + 300/1300 * 0.9340 = 0.9791$
    
- Gini:
    - $impurity_{split_1} = (550/1000) * (450/1000) + (450/1000) * (550/1000) = 0.495 $
    - $impurity_{split_2} = (105/300) * (195/300) + (195/300) * (105/300) = 0.455 $
    - $impurity_{total} = 1000/1300 * 0.495 + 300/1300 * 0.455 = 0.4857$
  
#### Impurity for Regression 

Since in regression we're dealing with continuous values, both entropy and gini are not applicable. To overcome this we can use other definitions:


- Variance: $impurity(\vec{y}) = variance (\vec{y}) = \frac{1}{N} * \sum \limits _{i=1}^{n} (y_i - \hat{y})^2 $


- MAE: $impurity(\vec{y}) = mae (\vec{y}) = \frac{1}{N} * \sum \limits _{i=1}^{n} |y_i - \hat{y}| $


#### Information Gain

Information gain is defined as the difference between the impurity of the parent node and the weighted sum of the child node impurities. The lower the impurity of the child nodes, the larger the information gain. We want fo focus on those splits that bring a high amount of information gain


#### Gini impurity and Regression

The Gini impurity is very close to the notion of mean squared error.


### CART Algorithm

CART stands for Classification and Regression Tress, it constructs a binary tree.

Loop over every feature $j$ and split every value it takes $x_{i,j}$:

- 1. Split the dataset into two parts (left and right):
    - $X_l, \vec{y_l}$ for data points $\vec{x_k}$ where $x_{k,j} < x_{i,j}$
    - $X_r, \vec{y_r}$ for data points $\vec{x_k}$ where $x_{k,j} >= x_{i,j}$
    - Let them have size $N_l$ and $N_r$
- 2. Keep track of the split that maximizes the average decrease in impurity (information gain)

- 3. Recursively pass the left and right datasets to the child nodes

- 4. If some stopping criteria is met, do nothing, return either the average or the most common class


Often times you want to sort your datapoints before computing the splits