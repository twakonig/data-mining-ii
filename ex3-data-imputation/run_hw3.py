"""
Course  : Data Mining II (636-0019-00L)
Theresa Wakonig
"""
from utils import *
from pca import *
from impute import *

import scipy.misc as misc
import scipy.ndimage as sn
import numpy as np 
from sklearn.datasets import make_moons
import matplotlib
import matplotlib.pyplot as plt
import seaborn_image as isns
'''
Main Function
'''
if __name__ in "__main__":
    
    # font size in plots:
    fs = 10
    matplotlib.rcParams['font.size'] = fs
    
    #################
    #Exercise 1:
    ranks = np.arange(1,31) #30

    # get image data:
    # shape: 512 x 512 -> m = n
    img = misc.ascent()
    X = sn.rotate(img, 180) #we rotate it for displaying with seaborn

    #generate data matrix with 60% missing values
    X_missing = randomMissingValues(X,per=0.60)

    #plot data for comparison
    fig, ax = plt.subplots(1, 2)
    isns.imgplot(X, ax=ax[0], cbar=False)

    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    ax[0].set_title('Original')
    ax[1].set_title(f'60% missing data')
    plt.savefig("exercise1_1.pdf")

    #Impute data with optimal rank r
    [X_imputed,r,testing_errors] = svd_imputation_optimised(
        X=X_missing,
        ranks=ranks,
        test_size=0.3
    )

    #plot data for comparison
    fig, ax = plt.subplots(1, 3)
    isns.imgplot(X, ax=ax[0], cbar=False)
    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    isns.imgplot(X_imputed, ax=ax[2], cbar=False)
    ax[0].set_title('Original', fontsize=fs)
    ax[1].set_title(f'60% missing data', fontsize=fs)
    ax[2].set_title('Imputed', fontsize=fs)
    plt.savefig("exercise1_2.pdf")

    #Plot testing_error and highlight optimial rank r
    plt.figure()
    plt.plot(ranks, testing_errors)
    plt.plot(int(r), testing_errors[int(r)-1], marker='o', color='r')
    plt.xlabel("rank r")
    pl.ylabel("MSE")
    pl.grid(True)
    # Save file
    plt.savefig('exercise1_3.pdf')

    #-----------------------plots for 1c)-----------------------------------
    X_imputed_s = svd_imputation(X_missing, 5)
    X_imputed_l = svd_imputation(X_missing, 29)

    # plot data for comparison
    fig, ax = plt.subplots(1, 3)
    isns.imgplot(X, ax=ax[0], cbar=False)
    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    isns.imgplot(X_imputed_s, ax=ax[2], cbar=False)
    ax[0].set_title('Original', fontsize=fs)
    ax[1].set_title(f'60% missing data', fontsize=fs)
    # r = 5
    ax[2].set_title('Imputed (too small r)', fontsize=fs)
    plt.savefig("exercise1_4.pdf")

    # plot data for comparison
    fig, ax = plt.subplots(1, 3)
    isns.imgplot(X, ax=ax[0], cbar=False)
    isns.imgplot(X_missing, ax=ax[1], cbar=False)
    isns.imgplot(X_imputed_l, ax=ax[2], cbar=False)
    ax[0].set_title('Original', fontsize=fs)
    ax[1].set_title(f'60% missing data', fontsize=fs)
    # r = 29
    ax[2].set_title('Imputed (too large r)', fontsize=fs)
    plt.savefig("exercise1_5.pdf")

    #-------------------------------------------------EX2----------------------------------------------------------
    #Exercise 2
    #load data
    # X...data points(300 x 2), y...classes of the 300 data points {0, 1}
    [X,y] = make_moons(n_samples=300,noise=None)

    #Perform a PCA
    #1. Compute covariance matrix
    Sigma = computeCov(X)

    #2. Compute PCA by computing eigen values and eigen vectors
    lambdas, principal_comps = computePCA(Sigma)

    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    X_centered = zeroMean(X)
    pca_compr = np.transpose(principal_comps[:, 0:2])
    X_red = transformData(pca_compr, X_centered)

    #4. Plot your transformed data and highlight the sample classes.
    plotTransformedData(X_red, y, "ex2a.pdf")

    #5. How much variance can be explained with each principle component?
    np.set_printoptions(precision=2)
    var = computeVarianceExplained(lambdas)
    print("Variance Explained Exercise 2a: ")
    for i in range(var.shape[0]):
        print("PC %d: %.2f"%(i+1,var[i]))


    #1. Perform Kernel PCA
    # transformed data: n_components x n_samples(300), here: 2 x 300
    transformed = np.array(RBFKernelPCA(X))

    #2. Plot your transformed data and highlight the sample classes
    plotTransformedData(transformed, y, "ex2b.pdf")

    #3. Repeat the previous 2 steps for gammas [1,5,10,20] and compare the results.
    gammas = np.array([1, 5, 10, 20])
    for gamma in gammas:
        trf = np.array(RBFKernelPCA(X, gamma, 2))
        plotTransformedData(trf, y, "ex2c_gamma{0}.pdf".format(gamma))
