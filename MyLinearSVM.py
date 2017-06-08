import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn.linear_model 
import sklearn.preprocessing



# This function computes the gradient of the objective funtion
def computegrad(beta, lambduh, x, y):
    # Calculate the maximum part of the gradient function
    innerMax = np.maximum(0,(1 - y*np.dot(x, beta)))
    # Calculate yx. Not a dot product.
    yx = y[:, None]*x
    # Get the n, number of examples
    n = x.shape[0]
    
    # Calculate the left hand side of the gradient formula
    lhs = yx*innerMax[:,None]
    
    # Sum within rows
    lhsSum = np.sum(lhs, axis=0)
    
    # Put everything together for the complete gradient formula
    grad = (-2*lhsSum/n) + 2*lambduh*beta
    
    return grad



# Calculate the objective function of the Linear SVM algorithm
def objective (beta , lambduh , x, y):
    # Return the value the objective function of the Linear SVM algorithm
    return 1/len(y) * np.sum(np.maximum(0,(1 - y*np.dot(x, beta)))**2) + lambduh*np.linalg.norm(beta)**2

# Finds step size using backtracking algorithm. Credit to Corrine Jones (TA).
def bt_line_search(beta, lambduh, x, y, eta=1, alpha=0.5, betaparam=0.8, maxiter=100):
    grad_beta = computegrad(beta, lambduh, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_eta = 0
    iter = 0
    while found_eta == 0 and iter < maxiter:
        if objective(beta - eta * grad_beta, lambduh, x=x, y=y) < objective(beta, lambduh, x=x, y=y)                 - alpha * eta * norm_grad_beta ** 2:
            found_eta = 1
        elif iter == maxiter - 1:
            print('Warning: Max number of iterations of backtracking line search reached')
        else:
            eta *= betaparam
            iter += 1
    return eta



# Implements Linear SVM using Fast Gradient optimization
def mylinearsvm(eta_init, maxiter, x, y,lambduh):
    # Get d, the number of features
    d = x.shape[1]
    
    #Initialize betas and thetas to an 0 zero array of length d
    beta = np.zeros(d) 
    theta = np.zeros(d) 
    
    # Get the gradient of theta
    grad_theta = computegrad(theta, lambduh, x=x, y=y)
    
    # Save the initial values of betas and thetas
    beta_vals = beta
    theta_vals = theta
    
    # Set initial iteration counter to 0
    iter = 0

    # Run this loop for maxiter times for the Fast Gradient Descent (FGD) Iterations
    while iter < maxiter:
        
        # Get step size using backtracking algorithm
        eta = bt_line_search(theta, lambduh, x=x, y=y, eta=eta_init)
        
        # Update beta_new and theta per FGD
        beta_new = theta - eta*grad_theta
        theta = beta_new + iter/(iter+3)*(beta_new-beta)

        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta_new))
        theta_vals = np.vstack((theta_vals, theta))
        
        # Get new gradient of theta
        grad_theta = computegrad(theta, lambduh, x=x, y=y)
        
        # Update beta
        beta = beta_new
        
        # Increase iteration counter
        iter += 1
        
        # Print out iteration counter at the 100s. Just to know things are working.
        if iter % 100 == 0:
            print('Fast gradient iteration', iter)
            
    # Return all values of betas and thetas. 
    # Use the last one for predictions.
    return beta_vals, theta_vals
    

# Calculates the MisClassification Error
def calcMisClassificationError(beta, x, y):

    # Get the predition
    pred = x.dot(beta)

    # Check accuracy and calculate error percent. Less than 0 is incorrect.
    check = y*pred
    error = 0
    for i in range(0,len(y)):
        if check[i] <0 :
            error = error + 1
    return error/len(y)*100


# Plot training progress
def showTrainingProgress(myBetas, lambduh, x_train, y_train):
    lenBeta = myBetas.shape[0]
    objValue = np.zeros(lenBeta)
    for i in range(lenBeta):
        objValue[i] = objective(myBetas[i], lambduh, x_train, y_train)

    fig, ax = plt.subplots()
    ax.plot(objValue, c='red')
    # ax.plot(mymce_test, c='blue', label='Test')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    # ax.legend(loc='upper right')
    plt.show()