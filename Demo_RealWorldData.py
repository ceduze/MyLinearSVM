import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing

# Lets import the code
from MyLinearSVM import *

# Pull in some real-world data and the corresponding train/test labels
spam = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
test_indicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ', header=None)

# Split data into x (features) and y (labels)
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1

# Convert to +/- 1
test_indicator = np.array(test_indicator).T[0]

# Divide the data into train, test sets
x_train = x[test_indicator == 0, :]
x_test = x[test_indicator == 1, :]
y_train = y[test_indicator == 0]
y_test = y[test_indicator == 1]

# Standardize the data.
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Set lambda and Call the SVM function t train data
lambduh = 1
myBetas, myThetas = mylinearsvm(0.0001, 1000, x_train, y_train,lambduh)

# Let's show the training progress
showTrainingProgress(myBetas, lambduh, x_train, y_train)

# Let's get the misclassification error on the train set
myMisCalc = calcMisClassificationError(myBetas[-1],x_train, y_train )
print("Misclassification error on train set:",myMisCalc)

# Let's get the misclassification error on the test set
myMisCalc = calcMisClassificationError(myBetas[-1],x_test, y_test )
print("Misclassification error on test set:",myMisCalc)