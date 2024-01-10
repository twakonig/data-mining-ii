"""
Homework 6: Transductive Support Vector Machines
Course  : Data Mining II (636-0019-00L)
Theresa Wakonig

Transductive linear SVM.
This file implements Transductive linear SVM described in the following paper:

Thorsten Joachims. Transductive Inference for Text Classification using Support Vector Machines. ICML 1999.
"""

import svm_linear
import numpy as np
import cvxopt
import cvxopt.solvers
from sklearn.metrics.pairwise import linear_kernel

'''
Checks condition to enter inner loop
Input: 
        y2
        slack2
Output:
        3-tuple: boolean whether condition is fulfilled, indices of samples where condition True
'''


def loopCondition(y2, slack2):
    for r in range(y2.shape[0]):
        for c in range(r, y2.shape[0]):
            # condition or inner while loop
            if (y2[r] * y2[c] < 0) and (slack2[r] > 0) and (slack2[c] > 0) and (slack2[r] + slack2[c] > 2.001):
                return True, r, c
    # condition did not evaluate to True
    return False, np.inf, np.inf


'''
Train linear transductive SVM
Input: 
        X1, matrix of size #sample1 x #feature (training data) (9930 features each)
        y1, vector of size #sample1, either -1 or 1
        X2, matrix of size #sample2 x #feature (testing data)
        C1, regularizer parameter, control the slack variable of samples from X1
        C2, regularizer parameter, control the slack variable of samples from X2
        p,  percentage of positive samples in unlabeled data X2, [0, 1]
Output:
        w, weight vector of size #feature
        b, intercept, a float
'''


def train(X1, y1, X2, C1, C2, p):
    # 1. Get number of positive samples
    num_pos = int(X2.shape[0] * p)

    # 2. Train standard SVM using labeled samples (inductive SVM trained on training data)
    w, b = svm_linear.train(X1, y1, C1)

    # 3. The num_pos test examples from X2 with highest value of w*x+b are assigned to 1
    # The rest of examples from X2 are assigned to -1
    eval_weights = np.zeros(X2.shape[0])
    for i in range(X2.shape[0]):
        # evaluate w*x+b for all samples
        eval_weights[i] = np.dot(w, X2[i][:]) + b

    # make predictions -> assign labels to test data
    y2 = np.full((X2.shape[0], 1), -1)
    max_indices = (-eval_weights).argsort()[:num_pos]
    y2[max_indices] = 1

    # 4. Retrain with label switching
    C_neg = 1e-5
    C_pos = 1e-5 * num_pos / (X2.shape[0] - num_pos)

    while C_neg < C2 or C_pos < C2:
        # 5. Retrain the variant of SVM
        w, b, slack1, slack2 = svm_linear.train_variant(X1, y1, X2, y2, C1, C_neg, C_pos)

        # 6. Take a positive and negative example, switch their labels (inner loop)
        condition, m, l = loopCondition(y2, slack2)
        while condition:
            # switch labels
            y2[m] *= -1
            y2[l] *= -1
            w, b, slack1, slack2 = svm_linear.train_variant(X1, y1, X2, y2, C1, C_neg, C_pos)
            condition, m, l = loopCondition(y2, slack2)

        # 7. Increase the value of C_neg and C_pos
        C_neg = min(2 * C_neg, C2)
        C_pos = min(2 * C_pos, C2)

    # 8. Return the learned model
    return w, b
