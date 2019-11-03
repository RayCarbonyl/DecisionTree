import numpy as np
import pandas as pd
from tree import*

#We are using data from UCI Machine Learning Repository for testing the classifier.
#archive.ics.uci.edu/ml/datasets/Iris
#archive.ics.uci.edu/ml/datasets/Car+Evaluation
iris_data = pd.read_csv("iris.data",header = None)
iris_data = iris_data.sample(frac=1).reset_index(drop=True)
iris_data.columns = ["sepal length","sepal width","petal length","petal width","class"]
iris_data
iris_x = iris_data.iloc[:100,:-1]
iris_y = iris_data.iloc[:100,-1]
iris_test_x = iris_data.iloc[100:,:-1]
iris_test_y = iris_data.iloc[100:,-1]

car_data = pd.read_csv("car.data",header = None)
car_data = car_data.sample(frac=1).reset_index(drop=True)
car_data.columns = ["buying","maint","doors","persons","lug_boot","safety","class"]
car_x = car_data.iloc[:1400,:-1]
car_y = car_data.iloc[:1400,-1]
car_test_x = car_data.iloc[1400:,:-1]
car_test_y = car_data.iloc[1400:,-1]



for criterion in ['entropy','gini']:
    for method in [None,'pre','post']:
        clf = Classifier(criterion,method)
        clf.fit(iris_x,iris_y,split = True,split_ratio = 0.7)
        acc = clf.eval(iris_test_x,iris_test_y)
        print(criterion,method,'pruning',acc,"acc on iris dataset")
        clf.plotting(clf.root)
        
        clf = Classifier(criterion,method)
        clf.fit(car_x,car_y,split = True,split_ratio = 0.7)
        acc = clf.eval(car_test_x,car_test_y)
        print(criterion,method,'pruning',acc,"acc on car dataset")
        clf.plotting(clf.root)