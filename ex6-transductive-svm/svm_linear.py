"""
Homework 6: Transductive Support Vector Machines
Course  : Data Mining II (636-0019-00L)
Theresa Wakonig

Linear SVM.
This file implements linear SVM and a variant of it by using a quadratic programming solver from cvxopt package
"""

import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.metrics.pairwise import linear_kernel

# Parts of source code SVM functions was extracted from:
# https://gist.github.com/mblondel/586753#file-svm-py
#
# Copyright (c) September 2010, Mathieu Blondel
# License: BSD 3 clause

'''
Train linear SVM
Input: 
        X, matrix of size #sample x #feature
        y, vector of size #sample, either -1 or 1
        C, regularizer parameter (default None for hard margin case)
Output:
        w, weight vector of size #feature
        b, intercept, a float
'''
def train(X, y, C=None):
    n_samples, n_features = X.shape

    # Compute Gram matrix
    K = linear_kernel(X)

    # Solve dual quadratic programming problem
    # min{0.5*a.T*P*a+q.T*a}
    # subject to G*a<=h and A*a=b
    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1,n_samples))
    b = cvxopt.matrix(0.0)
    if C is None:
        # Hard margin with constraint
        # ai >= 0
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
        # Soft margin with constraint
        # 0 <= ai <= C
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # Solve the quadratic programming problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = np.ravel(solution['x'])

    # Support vectors have non zero Lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    a = a[sv]
    sv_y = y[sv]
    sv_X = X[sv]

    # Calculate intercept b
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n],sv])
        b /= len(a)

    # Calculate weight vector w
    w = np.zeros(n_features)
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_X[n]

    return w, b


'''
Train a variant of linear SVM
Input: 
        X1, matrix of size #sample1 x #feature (5 x 9930)
        y1, vector of size #sample1, either -1 or 1
        X2, matrix of size #sample2 x #feature (5 x 9930)
        y2, vector of size #sample2, either -1 or 1
        C1, regularizer parameter, control the slack variable of samples from X1 (training points)
        C2, regularizer parameter, control the slack variable of postive samples from X2 (test points positive)
        C3, regularizer parameter, control the slack variable of negative samples from X2 (test points negative)
Output:
        w, weight vector of size #feature
        b, intercept, a float
        slack1, slack variable of samples from X1, vector of size #sample1
        slack2, slack variable of samples from X2, vector of size #sample2
'''
def train_variant(X1, y1, X2, y2, C1, C2, C3):
	# Concatenate two data set
    # (10 x 9930)
    X = np.vstack([X1, X2])
    # (10 x 1)
    y = np.append(y1, y2)

    # Create indicator vector mark of size #sample (10), where
    # mark[i] = 0, sample i is from X1
    # mark[i] = 1, sample i is from X2 and yi = 1
    # mark[i] =-1, sample i is from X2 and yi = -1
    n_samples, n_features = X.shape
    mark = y.copy()
    # set entries that stem from X1 to 0 (training points)
    mark[:X1.shape[0]] = 0

    # Compute Gram matrix
    K = linear_kernel(X)

    # Solve dual quadratic programming problem
    # min{0.5*a.T*P*a+q.T*a}
    # subject to G*a<=h and A*a=b
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y, (1, n_samples))
    b = cvxopt.matrix(0.0)

    if C1 == C2 == C3 == 0:
        # Hard margin with constraint
        # ai >= 0
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))
    else:
        # Calculate G and h with constraint
        # for xi in X1, 0 <= ai <= C1
        # for xi in X2 and yi is 1,  0 <= ai <= C2
        # for xi in X2 and yi is -1, 0 <= ai <= C3
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples)
        for i in range(n_samples):
            # set C coefficients
            if mark[i] == 0:
                tmp2[i] *= C1
            if mark[i] == 1:
                tmp2[i] *= C2
            else:
                tmp2[i] *= C3
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # Solve the quadratic programming problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = np.ravel(solution['x'])

    # Support vectors have non zero Lagrange multipliers
    sv = a > 1e-5
    ind = np.arange(len(a))[sv]
    a = a[sv]
    sv_y = y[sv]
    sv_X = X[sv]

    # Caculate intercept b
    b = 0
    for n in range(len(a)):
        b += sv_y[n]
        b -= np.sum(a * sv_y * K[ind[n], sv])
        b /= len(a)

    # Calculate weight vector w
    w = np.zeros(n_features)
    for n in range(len(a)):
        w += a[n] * sv_y[n] * sv_X[n]

    # Get slack variable
    slack = np.zeros(n_samples)
    for n in range(len(a)):
        if abs(C1 - a[n]) < 1e-5 or abs(C2 - a[n]) < 1e-5 or abs(C3 - a[n]) < 1e-5:
            slack[ind[n]] = abs(sv_y[n] - (np.dot(sv_X[n], w) + b))

    return w, b, slack[:X1.shape[0]], slack[X1.shape[0]:]