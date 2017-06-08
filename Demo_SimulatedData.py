import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.linear_model
import sklearn.preprocessing

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Lets import the code
from MyLinearSVM import *


# Let's generate some simulated data
X,Y = make_classification(n_samples=100, n_features=20)

# Let's split into train/test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


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