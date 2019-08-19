import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools

import matplotlib.pyplot as plt

def compute_feature_distances(X, y, classes=None, size=(7, 3.3)):
    """
    compute the average feature distances 
    between each pair of classes.
    Parameters:
        X : 2darray of n_sample x n_feature size
        y : 1darray of n_sample
    Returns:
        confusion matrix 1 : of the distances
        confusion matrix 2 : minimum non zero distance set to 1
        matplotlib fig : a plot of the confusion matrices
    """

    # first normalize the features
    X = StandardScaler().fit_transform(X)

    y_min = y.min()
    y -= y_min
    n_class = y.max() + 1
    if not classes:
        classes = [y_min + y for y in range(n_class)]

    cm = np.zeros((n_class, n_class))

    min_ = 1 << 31 - 1
    for i, j in itertools.product(range(n_class), range(n_class)):
        if i < j: 
            xi = X[y == i].mean(axis=0)
            xj = X[y == j].mean(axis=0)
            d = np.sqrt ( ((xi - xj) ** 2).sum() )
            cm[i, j] = cm[j, i] = d
            if d < min_: min_ = d
    
    cm2 = cm / min_

    # create a figure
    fig, axes = plt.subplots(1, 2, figsize=size, gridspec_kw = {'width_ratios':[1, 1]})

    plt.sca(axes[0])
    
    plt.imshow(cm, interpolation='nearest' , cmap=plt.get_cmap('Blues'))
    plt.title('Class Distances')
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(n_class), range(n_class)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.sca(axes[1])
    
    plt.imshow(cm, interpolation='nearest' , cmap=plt.get_cmap('Blues'))
    plt.title('Class Distances')
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f'
    thresh = cm2.max() / 2.
    for i, j in itertools.product(range(n_class), range(n_class)):
        plt.text(j, i, format(cm2[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm2[i, j] > thresh else "black")

    fig.tight_layout()

    return cm, cm2, fig


def save_data_svmlight_format(filepath, X, y, start_feature=0):
    """
    Save data from features and label in a file
    to match svmlight format:
        1) labels start at 1, so y + 1
        2) following features by index:value
           features are saved from 1 index
           X should be removed of the bias feature
           or provided start_feature
        3) if X and y are lists of numpy array, then a blank line is
           printed after each sequence
    Parameters:
        filepath : string - filepath to save the data
        X : numpy 2d-array or list of so
        y : numpy 1d-array or list of so
        start_feature : int - index to start saving features from
            saveing index starts from 1, adjusted column index accordingly
    """

    with open(filepath, 'wt') as f:
        if isinstance(X, list):
            for seq_x, seq_y in zip(X, y):
                for r in range(len(seq_y)):
                    line = " ".join( [str(seq_y[r])] + ["%d:%.8f"%(i+1, v)
                         for i,v in enumerate(seq_x[r, start_feature:]) ] ) + "\n"
                    f.write(line)
                # new line to separate sequences
                f.write("\n")
        else:
            for r in range(len(y)):
                line = " ".join( [str(y[r])] + ["%d:%.8f"%(i+1, v)
                        for i,v in enumerate(X[r, start_feature:]) ] ) + "\n"
                f.write(line)


def bayes_optimal_prediction(y_probs, cost_matrix):
    """
    Given a matrix of y-probs per sample, and 
    a cost_matrix , return the bayes optimal class
    i.e. argmax( cost_matrix * y_probs ) 
    Parameters:
        y_probs : n_smaple x n_class numpy array
        cost_matrix : n_class x n_class numpy array
    Returns:
        y_labels : n_sample 1d array
    """
    costs = np.dot(cost_matrix, y_probs.T)
    return costs.argmin(axis=0) 



if "__main__" == __name__:
    # X = np.random.random((10, 10))
    # y = np.array([1, 2, 3, 1, 1, 1, 2, 2, 2, 3])
    # cm1, cm2, fig = compute_feature_distances(X, y)
    # print (cm1)
    # print (cm2)
    # plt.show()
    # fig.savefig('cost_matrices.png')
    # start_feature = 1
    # row = 1
    # from sklearn.preprocessing import PolynomialFeatures
    # X = PolynomialFeatures().fit_transform(X)
    # line = " ".join( [str(y[row])] + ["%d:%.8f"%(i+1, v) for i,v in enumerate(X[row, start_feature:])] )
    # print (line)
    
    # y_prob = np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]])
    # cost_matrix = np.array([[0, 2], [1, 0]])
    # print (bayes_optimal_prediction(y_prob, cost_matrix))
    # y = np.array([1, 2, 1, 2])
    # y_bar = np.array([1, 2, 1, 1])

    # print (cost_matrix[y_bar-1, y-1].sum())

    pass