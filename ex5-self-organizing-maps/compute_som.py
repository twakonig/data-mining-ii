"""
Theresa Wakonig
HW5
"""
import random

from somutils import *
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

'''
Main Function
'''
if __name__ in "__main__":

    # Set up the parsing of command-line arguments
    parser = argparse.ArgumentParser(
        description="Compute SOM from data"
    )
    parser.add_argument(
        "--exercise",
        required=True,
        help="Indicator of exercise that will be solved. Either 1 or 2."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Path to directory where plots will be created"
    )
    parser.add_argument(
        "--p",
        required=True,
        help="Number of rows in the grid"
    )
    parser.add_argument(
        "--q",
        required=True,
        help="Number of columns in the grid"
    )
    parser.add_argument(
        "--N",
        required=True,
        help="Number of iterations"
    )
    parser.add_argument(
        "--alpha_max",
        required=True,
        help="Upper limit for learning rate"
    )
    parser.add_argument(
        "--epsilon_max",
        required=True,
        help="Upper limit for radius"
    )
    parser.add_argument(
        "--lamb",
        required=True,
        help="Decay constant for learning rate decay"
    )
    parser.add_argument(
        "--file",
        required=False,
        help="Full path to input file in Ex2"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # set random seed
    np.random.seed(0)

    # store command-line input
    p = int(args.p)
    q = int(args.q)
    N = int(args.N)
    alpha_max = int(args.alpha_max)
    epsilon_max = int(args.epsilon_max)
    lamb = float(args.lamb)

    # compute error in exercise 1
    comp_err = True

    if int(args.exercise) == 1:
        # ----------------------EXERCISE 1---------------------------------------
        print("Computing exercise 1.")

        # 1100 datapoints, 2D (1100 x 2 matrix)
        S = makeSCurve()

        buttons, grid, error = SOM(S, p, q, N, alpha_max, epsilon_max, comp_err, lamb)

        # plotting the data (S curve) and buttons
        output_b = os.path.join(args.outdir, "exercise_1b.pdf")
        plotDataAndSOM(S, buttons, output_b)

        if comp_err:
                output_c = os.path.join(args.outdir, "exercise_1c.pdf")
                t = np.arange(1, N + 1)
                plt.figure()
                plt.plot(t, error)
                plt.title('Reconstruction error for the SOM as a function of iteration t')
                plt.xlabel("iteration t")
                plt.ylabel("reconstruction error")
                plt.grid(True)
                plt.savefig(output_c)
    else:
        # ----------------------EXERCISE 2---------------------------------------
        print("Computing exercise 2.")
        input_file = args.file

        # columns 3 to 7 as data matrix X
        X = np.loadtxt(input_file, delimiter=",", skiprows=1, usecols=(3,4,5,6,7))
        crab_data = np.loadtxt(input_file, dtype=str, delimiter=",", skiprows=1, usecols=(0,1,2))

        # subscript c for "crab"
        buttons_c, grid_c, error_c = SOM(X, p, q, N, alpha_max, epsilon_max, False, lamb)
        labels = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            labels[i] = findNearestButtonIndex(X[i][:], buttons_c)

        # ------------------------------------Write labels in text file -> Ex. 2a)-------------------------------------
        try:
            file_name = "{}/output_som_crabs.txt".format(args.outdir)
            f_out = open(file_name, 'w')
        except IOError:
            print("Output file {} cannot be created".format("output_som_crabs.txt"))
            sys.exit(1)

        # Write header for output file
        f_out.write('{}\t{}\t\t{}\t{}\n'.format(
            'sp',
            'sex',
            'index',
            'label'))

        for i in range(X.shape[0]):
            f_out.write('{}\t{}\t\t{}\t\t{}\n'.format(
                crab_data[i][0],
                crab_data[i][1],
                crab_data[i][2],
                labels[i])
            )
        f_out.close()

        #-----------------------------------------------Ex. 2b)----------------------------------------------------
        output_2b = os.path.join(args.outdir, "exercise_2b.pdf")
        plotSOMCrabs(X, crab_data[:, 0:2], grid_c, p, q, buttons_c, output_2b)







