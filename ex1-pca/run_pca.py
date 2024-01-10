"""
Homework: Principal Component Analysis
Course  : Data Mining II (636-0019-00L)
"""
# Name: Theresa Wakonig

#import all necessary functions
from utils import *
from pca import *

'''
Main Function
'''
if __name__ in "__main__":
    #Initialise plotting defaults
    initPlotLib()

    ##################
    #Exercise 2:
    
    #Simulate Data
    # 320 x 40
    # object of type utils.Data: -> has four attributes: data, target, n_samples (rows), n_features (cols)
    # data.data -> feature matrix; data.target -> class labels
    data = simulateData()

    #Perform a PCA
    X = data.data
    # mean-center the x_i before computing covariance
    X_centered = X - np.mean(X, axis = 0, keepdims=True)
    assert(X_centered.shape[0] > X_centered.shape[1])

    # 1. Compute covariance matrix, 40 x 40 (covariance among features)
    Sigma = computeCov(X_centered)

    #2. Compute PCA by computing eigen values and eigen vectors
    lambdas, principal_comps = computePCA(Sigma)

    #3. Transform your input data onto a 2-dimensional subspace using the first two PCs
    # use compression matrix
    pca_compr = np.transpose(principal_comps[:, 0:2])
    print(f'dimensions of principal_comps (EV): {np.shape(principal_comps)}')
    X_red = transformData(pca_compr, X_centered)

    #4. Plot your transformed data and highlight the four different sample classes
    plotTransformedData(X_red, data.target)

    #5. How much variance can be explained with each principle component?
    expl_var = computeVarianceExplained(lambdas)
    alpha = np.cumsum(expl_var)
    idx_50 = np.min(np.where(alpha >= 0.50))
    idx_80 = np.min(np.where(alpha >= 0.80))
    idx_95 = np.min(np.where(alpha >= 0.95))

    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.1: ")
    for i in range(15):
         print("PC %d: %.2f"%(i+1,expl_var[i]))

    print("Principal components necessary to explain at least ... ")
    print(f'... 50%: {idx_50 + 1}')
    print(f'... 80%: {idx_80 + 1}')
    print(f'... 95%: {idx_95 + 1}')

    #6. Plot cumulative variance explained per PC
    plotCumSumVariance(expl_var)




    ##################
    #Exercise 2 Part 2:
    
    #1. normalise data
    X_normalized = dataNormalisation(X)

    #2. compute covariance matrix
    Sigma_2 = computeCov(X_normalized)

    #3. compute PCA
    lambdas_2, principal_comps_2 = computePCA(Sigma_2)

    #4. Transform your input data inot a 2-dimensional subspace using the first two PCs
    pca_compr_2 = np.transpose(principal_comps_2[:, 0:2])
    X_red_2 = transformData(pca_compr_2, X_normalized)

    #5. Plot your transformed data
    plotTransformedData(X_red_2, data.target)

    #6. Compute Variance Explained
    expl_var_2 = computeVarianceExplained(lambdas_2)
    alpha_2 = np.cumsum(expl_var_2)
    idx_50_2 = np.min(np.where(alpha_2 >= 0.50))
    idx_80_2 = np.min(np.where(alpha_2 >= 0.80))
    idx_95_2 = np.min(np.where(alpha_2 >= 0.95))

    np.set_printoptions(precision=2)
    print("Variance Explained Exercise 2.2: ")
    for i in range(15):
        print("PC %d: %.2f" % (i + 1, expl_var_2[i]))

    print("Principal components necessary to explain at least ... ")
    print(f'... 50%: {idx_50_2 + 1}')
    print(f'... 80%: {idx_80_2 + 1}')
    print(f'... 95%: {idx_95_2 + 1}')

    #7. Plot Cumulative Variance
    plotCumSumVariance(expl_var_2)
    