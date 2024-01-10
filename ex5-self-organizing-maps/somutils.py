"""
Theresa Wakonig
Homework: Self-organizing maps
Course  : Data Mining II (636-0019-00L)

Auxiliary functions to help in the implementation of an online version
of the self-organizing map (SOM) algorithm.
"""
# Author: Dean Bodenham, May 2016
# Modified by: Damian Roqueiro, May 2017
# Modified by: Christian Bock, April 2021

import scipy as sp
from sklearn import datasets
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import matplotlib.patches as mpatches
import numpy as np

"""
A function to create the S curve
"""


def makeSCurve():
    n_points = 1000
    noise = 0.2
    X, color = datasets.make_s_curve(n_points, noise=noise, random_state=0)
    Y = np.array([X[:, 0], X[:, 2]])
    Y = Y.T
    # Stretch in all directions
    Y = Y * 2

    # Now add some background noise
    xMin = np.min(Y[:, 0])
    xMax = np.max(Y[:, 0])
    yMin = np.min(Y[:, 1])
    yMax = np.max(Y[:, 1])

    # integer division (floor)
    n_bg = n_points // 10
    Ybg = np.zeros(shape=(n_bg, 2))
    Ybg[:, 0] = np.random.uniform(low=xMin, high=xMax, size=n_bg)
    Ybg[:, 1] = np.random.uniform(low=yMin, high=yMax, size=n_bg)

    Y = np.concatenate((Y, Ybg))
    return Y


"""
Plot the data and SOM for the S-curve
  data: 2 dimensional dataset (first two dimensions are plotted)
  buttons: N x 2 array of N buttons in 2D
  fileName: full path to the output file (figure) saved as .pdf or .png
"""


def plotDataAndSOM(data, buttons, fileName):
    fig = plt.figure(figsize=(8, 8))
    # Plot the data in grey
    plt.scatter(data[:, 0], data[:, 1], c='grey')
    # Plot the buttons in large red dots
    plt.plot(buttons[:, 0], buttons[:, 1], 'ro', markersize=10)
    # Label axes and figure
    plt.title('S curve dataset, with buttons in red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(fileName)


"""
Create a grid of points, dim p x q, and save grid in a (p*q, 2) array
  first column: x-coordinate
  second column: y-coordinate
"""


def createGrid(p, q):
    index = 0
    grid = np.zeros(shape=(p * q, 2))
    for i in range(p):
        for j in range(q):
            index = i * q + j
            grid[index, 0] = i
            grid[index, 1] = j
    return grid


"""
A function to plot the crabs results
It applies a SOM previously computed (parameters grid and buttons) to a given
dataset (parameters data)

Parameters
 X : is the original data that was used to compute the SOM.
     Rows are samples and columns are features.
 idInfo : contains the information (sp and sex for the crab dataset) about
          each data point in X.
          The rows in idInfo match one-to-one to rows in X.
 grid, buttons : obtained from computing the SOM on X.
 fileName : full path to the output file (figure) saved as .pdf or .png
    # Blue male:     dark blue #0038ff
    # Blue female:   cyan      #00eefd
    # Orange male:   orange    #ffa22f
    # Orange female: yellow    #e9e824
"""


def plotSOMCrabs(X, idInfo, grid, p, q, buttons, fileName):
    # put labels and colors into array
    label_names = ['Blue male', 'Blue female', 'Orange male', 'Orange female']
    colors = ['#0038ff', '#00eefd', '#ffa22f', '#e9e824']

    # parameter for random jitter (normal distribution)
    sigma = 0.008

    # Find coordinates in 2D equally spaced grid -> according to nearest button
    grid_ids = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        label = findNearestButtonIndex(X[i][:], buttons)
        # compute ID of point i in grid (p x q grid (6 x 8)) AND ADD random jitter
        # row index
        grid_ids[i][0] = label // q + np.random.normal(0, sigma, 1)
        # column index
        grid_ids[i][1] = label - grid_ids[i][0] * q + np.random.normal(0, sigma, 1)

    # prepare patches for legend
    handles = []
    for color in colors:
        handles.append(mpatches.Rectangle((0, 0), 1, 1, fc=color))

    # define color of each data point (class it belongs to)
    # four combinations of species and sex -> four colors
    color_code = np.array(['#AAAAAA'] * X.shape[0])
    species_sex = [['"B"', '"M"'], ['"B"', '"F"'], ['"O"', '"M"'], ['"O"', '"F"']]
    for i, type in enumerate(species_sex):
        index = np.where((idInfo[:, 0] == type[0]) & (idInfo[:, 1] == type[1]))
        color_code[index] = colors[i]

    # ----------------------------plot figure--------------------------------
    plt.figure(figsize=(12, 10))
    # draw circles with centers at grid points (p x q grid (6 x 8))
    plt.scatter(grid[:, 1], grid[:, 0], s=1300, facecolors='none', edgecolors='k')
    # plot points according to their grid_ids (coordinates in grid)
    plt.scatter(grid_ids[:, 1], grid_ids[:, 0], s=10, c=color_code)
    plt.title("SOM of crabs data (on 6x8 grid)")
    # adding the legend
    plt.legend(handles, label_names, loc='best', bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()
    plt.savefig(fileName)


"""
Function for computing distance in grid space.
Use Euclidean distance.
"""


def getGridDist(z0, z1):
    # returns Euclidean distance
    return np.linalg.norm(z0 - z1)


"""
Function for computing distance in feature space.
Use Euclidean distance.
"""


def getFeatureDist(z0, z1):
    # feature space with d dimensions: input vectors of size d x 1 -> returns Euclidean distance
    return np.linalg.norm(z0 - z1)


"""
Create distance matrix between points numbered 1,2,...,K=p*q from grid
"""


def createGridDistMatrix(grid):
    # pairwise distance matrix D
    d = distance.pdist(grid, 'euclidean')
    D = distance.squareform(d)
    return D


"""
Create array for epsilon. Values in the array decrease to 1.
"""


def createEpsilonArray(epsilon_max, N):
    return np.linspace(epsilon_max, 1, N)


"""
Create array for alpha. Values in the array decrease from 
alpha_max according to the equation in the homework sheet.
"""


def createAlphaArray(alpha_max, lambda_, N):
    t = np.linspace(0, N, N)
    return alpha_max * np.exp(-lambda_ * t)


"""
X is whole data set, K is number of buttons to choose
"""


# buttons in feature space F (randomly chosen)
def initButtons(X, K):
    n = X.shape[0]
    # Kx1 vector of random IDs
    rand_id = np.random.choice(n, size=K, replace=False)
    B = X[rand_id][:]
    return B


"""
x is one data point, buttons is the grid in FEATURE SPACE
"""


def findNearestButtonIndex(x, buttons):
    K = buttons.shape[0]
    min_dist = np.inf
    star_id = 0

    for i in range(K):
        dist = getFeatureDist(x, buttons[i][:])
        if (dist < min_dist):
            min_dist = dist
            star_id = i
    return star_id


"""
Find all buttons within a neighborhood of epsilon of index IN GRID SPACE 
(return a boolean vector)
"""


def findButtonsInNhd(index, epsilon, buttonDist):
    bool_ids = (buttonDist[index, :] < epsilon)
    return bool_ids


"""
Do gradient descent step, update each button position IN FEATURE SPACE
"""


def updateButtonPosition(button, x, alpha):
    button_new = button + alpha * (x - button)
    return button_new


"""
Compute the squared distance between data points and their nearest button
"""


def computeError(data, buttons):
    n = data.shape[0]
    reconstr_err = 0
    # go over all points
    for i in range(n):
        min_id = findNearestButtonIndex(data[i][:], buttons)
        reconstr_err += (getFeatureDist(data[i][:], buttons[min_id][:]))**2
    # return reconstruction error
    return reconstr_err


"""
Implementation of the self-organizing map (SOM)

Parameters
 X : data, rows are samples and columns are features
 p, q : dimensions of the grid
 N : number of iterations
 alpha_max : upper limit for learning rate
 epsilon_max : upper limit for radius
 compute_error : boolean flag to determine if the error is computed.
                 The computation of the error is time-consuming and may
                 not be necessary every time the function is called.
 lambda_ : decay constant for learning rate
                 
Returns
 buttons, grid : the buttons and grid of the newly created SOM
 error : a vector with error values. This vector will contain zeros if 
         compute_error is False

TODO: Complete the missing parts in this function following the pseudocode
      in the homework sheet
"""


def SOM(X, p, q, N, alpha_max, epsilon_max, compute_error, lambda_):
    # 1. Create grid and compute pairwise distances
    grid = createGrid(p, q)
    gridDistMatrix = createGridDistMatrix(grid)

    # 2. Randomly select K out of d data points as initial positions of the buttons
    K = p * q
    # number of data points n
    n = X.shape[0]
    # Kx2 vector of initial prototype coordinates in feature space F (m_1,..., m_K)
    buttons = initButtons(X, K)

    # 3. Create a vector of size N for learning rate alpha
    alpha_vec = createAlphaArray(alpha_max, lambda_, N)

    # 4. Create a vector of size N for epsilon
    epsilon_vec = createEpsilonArray(epsilon_max, N)

    # Initialize a vector with N zeros for the error
    # This vector may be returned empty if compute_error is False
    error_vec = np.zeros(N)

    # 5. Iterate N times
    for i in range(N):
        # 6. Initialize/update alpha and epsilon
        alpha = alpha_vec[i]
        epsilon = epsilon_vec[i]

        # 7. Choose a random index id in {1, 2, ..., n}
        rand_id = np.random.choice(n, size=1)

        # 8. Find button m_star that is nearest to x_id in F
        # randomly selected datapoint x_id. Want to find nearest prototype (m_star) in F
        x_id = np.array(X[rand_id][:])
        id_star = findNearestButtonIndex(x_id, buttons)

        # 9. Find all grid points in epsilon-neighbourhood of m_star in GRID SPACE G
        eps_nbh_gridspace = findButtonsInNhd(id_star, epsilon, gridDistMatrix)

        # # 10. Update position (in FEATURE SPACE) of all buttons m_j in epsilon-nhd of m_star, including m_star
        for k in range(K):
            if eps_nbh_gridspace[k]:
                buttons[k] = updateButtonPosition(buttons[k], x_id, alpha)

        # Compute the error
        # Note: The computation takes place only if compute_error is True
        if compute_error:
            error_vec[i] = computeError(X, buttons)

    # 11. Return buttons, grid and error
    return buttons, grid, error_vec
