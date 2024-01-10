#import all necessary functions
from utils import *
from pca import *
from pinv import *

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:

    #Get Iris Data
    data = loadIrisData()
    
    #Perform a PCA using covariance matrix and eigen-value decomposition
    #1. Compute covariance matrix
    #2. Compute PCA by computing eigen values and eigen vectors
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    #4. Plot your transformed data and highlight the three different sample classes
    #5. How much variance can be explained with each principle component?
    var = []
    print("Variance Explained PCA: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))


    #Perform a PCA using SVD
    #1. Normalise data by substracting the mean
    #2. Compute PCA by computing eigen values and eigen vectors
    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    #4. Plot your transformed data and highlight the three different sample classes
    #5. How much variance can be explained with each principle component?
    var = []
    print("Variance Explained SVD: ")
    for i in range(var.shape[0]):
       print("PC %d: %.2f"%(i+1,var[i]))


    #Exercise 3
    #1. Compute the Moore-Penrose Pseudo-Inverse on the Iris data

    #2. Check Properties

    print("\nChecking status exercise 3:")
    status = False
    print(f"X X^+ X = X is {status}")
    status = False
    print(f"X^+ X X^+ = X^+ is {status}")
