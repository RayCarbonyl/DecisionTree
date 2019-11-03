# Decision Tree 
## Introduction
This is an implementation of desicion tree classifier and regressor.
### Classifier:
Criterion for selecting attribute/feature can be chosed among information gain,information gain ratio and gini index.The attributes/features of samples can be continuous or categorical.Pre-pruning and post-pruning method is available for avoiding overfitting.
### Regressor:
The regressor generates a binary regression tree from input samples.The goal of each selection of variable and value for splitting is to minimize the sum of squared error of two subregion after splitting.
## Usage
The tree classifier takes as input Pandas Dataframe x and Pandas series y.

The tree classifier takes as input two numpy ndarray x and y.

Both classifier and regressor provide plotting method for  visualization.

Check out test.py for examples of fitting data,predicting and  evaluating.

## Requirements
Python3

Numpy

Pandas
