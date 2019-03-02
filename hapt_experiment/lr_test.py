"""
Data set: http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions 

Trains a Logistric regression model, and 
predict using argmin( cost * prediction_probability )

run using:
    PYTHONPATH=<adv_path> python <data_dir> <cost_file>
        adv_path: path to directory containing AdversarialGame
        data_dir: contains Train/{X_train.txt,sequnce_splits_train.txt} 
                Test/{X_test.txt,sequnce_splits_test.txt}
        cost_file: csv file containing the cost matrix
"""

import os, sys

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

from analysis.utils import bayes_optimal_prediction


def load_hapt_data(data_dir = '/Users/kaiser/Downloads/Dataset/Sequence/HAPT Data Set/'):
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')

    X_file, y_file, seq_file = 'X_%s.txt', 'y_%s.txt', 'sequnce_splits_%s.txt'
    X_tr, y_tr, seq_tr = ( 
        np.loadtxt(os.path.join(train_dir, X_file%'train')),
        np.loadtxt(os.path.join(train_dir, y_file%'train')).astype(int),  
        np.loadtxt(os.path.join(train_dir, seq_file%'train'), delimiter=',').astype(int) 
    )
    X_ts, y_ts, seq_ts = ( 
        np.loadtxt(os.path.join(test_dir, X_file%'test')),
        np.loadtxt(os.path.join(test_dir, y_file%'test')).astype(int),  
        np.loadtxt(os.path.join(test_dir, seq_file%'test'), delimiter=',').astype(int) 
    )
        
    X_tr = [X_tr[seq[1]:seq[2]+1] for seq in seq_tr] # 0 based and end-inclusive split info
    y_tr = [y_tr[seq[1]:seq[2]+1] for seq in seq_tr]

    X_ts = [X_ts[seq[1]:seq[2]+1] for seq in seq_ts]
    y_ts = [y_ts[seq[1]:seq[2]+1] for seq in seq_ts]

    return X_tr, y_tr, X_ts, y_ts


def preprocess(X_tr, X_ts, poly_degree=1):
    """
    Do polynomial transform
    also return the combined transform, incase needed

    features are normalized already in the source
    so, only polynomial transformation is done
    default is 1, since 561 fetures is already too many
    """
    poly = preprocessing.PolynomialFeatures(degree=poly_degree, interaction_only=False)
    
    X_comb_tr = poly.fit_transform(np.concatenate(X_tr, axis=0))
    X_comb_ts = poly.transform(np.concatenate(X_ts, axis=0))

    X_tr = [poly.transform(x) for x in X_tr]
    X_ts = [poly.transform(x) for x in X_ts]

    return X_tr, X_ts, X_comb_tr, X_comb_ts


def grid_search(clf, X_tr, y_tr, val_idx):

    # extract the sequences to be used
    X_tr = [X_tr[i] for i in val_idx]
    y_tr = [y_tr[i] for i in val_idx]

    X_tr = np.concatenate(X_tr, axis=0)
    y_tr = np.concatenate(y_tr)

    param_grid = {'C' : (0.0001, 0.001, 0.01, 0.1, 1., 2., 10.) } 

    kfold = KFold(5, shuffle=False) # don't shuffle keep consistent splits across algorithms
    gs = GridSearchCV(clf, param_grid, cv=kfold.split(X_tr))
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def run(data_dir, cost_file):
    # load data, scale may not be needed
    X_tr, y_tr, X_ts, y_ts = load_hapt_data(data_dir)
    X_tr, X_ts, X_comb_tr, _ = preprocess(X_tr, X_ts)
    
    # load cost_matrix
    cost_matrix = np.loadtxt(cost_file, delimiter=',')

    # extract some random indices of 20% for grid search
    # val_idx = np.random.permutation(len(X_tr))
    # val_idx = val_idx[:len(X_tr) // 5]
    val_idx = np.loadtxt(os.path.join(data_dir, 'Train/validation_set.txt'), delimiter=',').astype(int)

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=3000)
    
    lr, best_param = grid_search(lr, X_tr, y_tr, val_idx)
    print ("best parameter: " + str (best_param) )

    lr.fit(X_comb_tr, np.concatenate(y_tr))

    # predict
    prediction_result = []
    total_cost, total_cost_bayes, total_length = 0.0, 0.0, 0
    with open('predictions.txt', 'wt') as f:
        for x, y in zip(X_ts, y_ts):
            y_probs = lr.predict_proba(x)
            y_pred = lr.predict(x)
            y_bayes = bayes_optimal_prediction(y_probs, cost_matrix) + 1

            f.write( "p_max:" + ",".join([str(i) for i in y_pred]) + "\n")
            f.write( "bayes:" + ",".join([str(i) for i in y_bayes]) + "\n")

            cost_pred = cost_matrix[y_pred-1, y-1].sum()
            cost_bayes = cost_matrix[y_bayes-1, y-1].sum()
            total_cost += cost_pred
            total_cost_bayes += cost_bayes
            total_length += len(y)
            prediction_result.append(
                "%d,%f,%f,%f,%f\n" % (
                    len(y),
                    cost_pred,
                    cost_pred / len(y),
                    cost_bayes,
                    cost_bayes / len(y)
                )
            )

    with open('prediction_result.txt', 'wt') as f:
        for line in prediction_result:
            f.write(line)

    print ("micro average: loss:", total_cost/total_length, "bayes-loss:", total_cost_bayes/total_length)
            

if "__main__" == __name__:
    Usage = "PYTHONPATH=<Adv lib parent> python <this file> <data dir> <cost file>"
    if len(sys.argv) < 3:
        print ("Usage:", Usage) 
        exit(0)

    data_dir = os.path.abspath(sys.argv[1])
    cost_file = os.path.abspath(sys.argv[2])
    run(data_dir, cost_file)
    