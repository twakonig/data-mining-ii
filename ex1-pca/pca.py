import scipy as sp
import scipy.linalg as linalg
import pylab as pl
import numpy as np

from utils import plot_color

# Name: Theresa Wakonig
'''############################'''
'''Principle Component Analyses'''
'''############################'''

'''
Compute Covariance Matrix
Input: Matrix of size #samples x #features
Output: Covariance Matrix of size #features x #features
Note: Do not use scipy or numpy cov. Implement the function yourself.
      You can of course add an assert to check your covariance function
      with those implemented in scipy/numpy.
'''
def computeCov(X):
    n = X.shape[0]
    assert (np.all(n**(-1) * np.transpose(X).dot(X)) == np.all(np.cov(X, None, False, True)))
    return n**(-1) * np.transpose(X).dot(X)


'''
Compute PCA
Input: Covariance Matrix
Output: [eigen_values,eigen_vectors] sorted in such a why that eigen_vectors[:,0] is the first principle component
        eigen_vectors[:,1] the second principle component etc...
Note: Do not use an already implemented PCA algorithm. However, you are allowed to use an implemented solver 
      to solve the eigenvalue problem!
'''
def computePCA(Sigma):
    eigenVals = linalg.eig(Sigma)[0]
    eigenVecs = linalg.eig(Sigma)[1]
    # sorted list of indices from biggest to smallest lambda
    sort = np.argsort(eigenVals)[::-1]
    eigenVals = eigenVals[sort]
    eigenVecs = eigenVecs[:, sort]
    return eigenVals, eigenVecs

'''
Transform Input Data Onto New Subspace
Input: pcs: matrix containing the first x principle components
       data: input data which should be transformed onto the new subspace
Output: transformed input data. Should now have the dimensions #samples x #components_used_for_transformation
'''
def transformData(W, X):
    print(f'dimensions of W: {np.shape(W)}')
    print(f'dimensions of X: {np.shape(X)}')
    compressed = np.dot(W, np.transpose(X))
    return np.transpose(compressed)

'''
Compute Variance Explaiend
Input: eigen_values
Output: return vector with varianced explained values. Hint: values should be between 0 and 1 and should sum up to 1.
'''
def computeVarianceExplained(eigenVals):
    p = eigenVals.shape[0]
    total_var = np.sum(eigenVals)
    alpha = (np.cumsum(eigenVals) / total_var).real
    shifted = np.append([0.0], alpha[0:p-1])
    return alpha - shifted


'''############################'''
'''Different Plotting Functions'''
'''############################'''

'''
Plot Transformed Data
Input: transformed: data matrix (#samples x 2)
       labels: target labels, class labels for the samples in data matrix
       filename: filename to store the plot
'''
def plotTransformedData(X_red, target,filename="exercise1.pdf"):
    pl.figure(figsize=(32, 20))
    labels = np.unique(target)
    #PLOT FIGURE HERE
    for i in range(X_red.shape[0]):
        pl.scatter(X_red[i, 0], X_red[i, 1], color = plot_color[int(target[i])], s = 150)
    pl.title("PCA", size = 35)
    pl.legend(labels, scatterpoints = 1, numpoints = 1, prop = {'size': 20}, ncol = 4, loc = "upper right", fancybox = True)
    pl.xlabel("Transformed PC1", size = 20)
    pl.ylabel("Transformed PC2", size = 20)
    pl.savefig(filename)

'''
Plot Cumulative Explained Variance
Input: var: variance explained vector
       filename: filename to store the file
'''
def plotCumSumVariance(var,filename="cumsum.pdf"):
    pl.figure()
    pl.plot(sp.arange(var.shape[0]),sp.cumsum(var)*100)
    pl.xlabel("Principle Component")
    pl.ylabel("Cumulative Variance Explained in %")
    pl.grid(True)
    #Save file
    pl.savefig(filename)



'''############################'''
'''Data Preprocessing Functions'''
'''############################'''

'''
Exercise 2 Part 2:
Data Normalisation (Zero Mean, Unit Variance)
'''
def dataNormalisation(X):
    X = X - np.mean(X, axis=0, keepdims=True)
    return X / X.std(axis = 0)
