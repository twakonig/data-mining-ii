"""
Homework 6: Transductive Support Vector Machines
Course  : Data Mining II (636-0019-00L)
Theresa Wakonig

Main program.
This file reads an example text data set (sparse tf-idf vectors) and evaluates 
the performance of inductive linear SVM and transductive linear SVM.
"""

#import all necessary functions
import numpy as np
import svm_linear
import tsvm_linear
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file

'''
Main Function
'''
if __name__ in "__main__":
    # Load Data
    test = load_svmlight_file('test.dat')
    train = load_svmlight_file('train.dat')
    test_X = np.array(test[0].todense())
    test_y = test[1]
    train_X = np.array(train[0].todense())
    train_y = train[1] 

    # Extract labeled and unlabeled training samples
    # 1, -1 indicate labeled samples
    # 0 indicates unlabeled samples
    train_label_X = train_X[np.where(train_y!=0)]
    train_label_y = train_y[np.where(train_y!=0)]
    train_unlabel_X = train_X[np.where(train_y==0)]

    # Perform inductive linear SVM
    print("Inductive SVM")
    w, b = svm_linear.train(train_label_X, train_label_y, C=1)
    pred_y = np.sign(np.dot(test_X, w) + b)
    # scores an accuracy of 0.855
    print("Accuracy: %s\n" % accuracy_score(test_y, pred_y))

    # Split the training set into two parts
    n = train_label_X.shape[0] // 2
    train_label_X1 = train_label_X[:n]
    train_label_y1 = train_label_y[:n]
    train_label_X2 = train_label_X[n:]
    train_label_y2 = train_label_y[n:]
    # Perform a variant of inductive linear SVM
    print("Inductive SVM variate")
    w, b, slack1, slack2 = svm_linear.train_variant(train_label_X1, train_label_y1, train_label_X2, train_label_y2,
                                                    C1=1, C2=1, C3=1)
    pred_y = np.sign(np.dot(test_X, w) + b)
    print("Accuracy: %s\n" % accuracy_score(test_y, pred_y))

    # ------------------------------------------EX2-------------------------------------------------

    # Perform transductive linear SVM
    print("Transductive SVM")
    w, b = tsvm_linear.train(train_label_X, train_label_y, train_unlabel_X, C1=1, C2=0.01, p=0.5)
    pred_y = np.sign(np.dot(test_X, w) + b)
    print("Accuracy: %s" % accuracy_score(test_y, pred_y))

    
