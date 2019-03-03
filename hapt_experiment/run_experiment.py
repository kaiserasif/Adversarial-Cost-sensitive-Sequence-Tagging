"""
Data set: http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions 

run using:
    PYTHONPATH=<adv_path> python <data_dir> <cost_file> <svm_data_save_dir>
        adv_path: path to directory containing AdversarialGame
        data_dir: contains Train/{X_train.txt,sequnce_splits_train.txt} 
                Test/{X_test.txt,sequnce_splits_test.txt}
        cost_file: csv file containing the cost matrix
        svm_data_save_dir: if provided, svm style features saved
"""

import os, sys

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold, GridSearchCV

from AdversarialGame.classifiers import CostSensitiveSequenceTagger

from analysis.utils import save_data_svmlight_format


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


def grid_search(clf, X_tr, y_seq, val_idx):

    # extract the sequences to be used
    X_tr = [X_tr[i] for i in val_idx]
    y_seq = [y_seq[i] for i in val_idx]

    param_grid = {'reg_constant' : (0.0001, 0.001, 0.01, 0.1, 1., 2., 10.), 
                    'learning_rate': (0.003, 0.01, 0.03, 0.1, 0.3) } 

    kfold = KFold(5, shuffle=False) # don't shuffle keep consistent splits across algorithms
    gs = GridSearchCV(clf, param_grid, cv=kfold.split(X_tr))
    gs.fit(X_tr, y_seq)
    return gs.best_estimator_, gs.best_params_


def run(data_dir, cost_file, svm_data_dir):
    # load data, scale may not be needed
    X_tr, y_tr, X_ts, y_ts = load_hapt_data(data_dir)
    X_tr, X_ts, _, _ = preprocess(X_tr, X_ts)
    y_seq = [y-1 for y in y_tr] # for ast, 0 indexed classes
    
    # load cost_matrix
    cost_matrix = np.loadtxt(cost_file, delimiter=',')

    reg_constant, learning_rate = 0.01, 0.01

    # extract some random indices of 20% for grid search
    # val_idx = np.random.permutation(len(X_tr))
    # val_idx = val_idx[:len(X_tr) // 5]
    val_idx = np.loadtxt(os.path.join(data_dir, 'Train/validation_set.txt'), delimiter=',').astype(int)

    # save for svm
    if svm_data_dir:
        save_data_svmlight_format(os.path.join(svm_data_dir, 'train.dat'), X_tr, y_tr, start_feature=1)
        save_data_svmlight_format(os.path.join(svm_data_dir, 'test.dat'), X_ts, y_ts, start_feature=1)
        
    
    # now create classifier and train
    ast = CostSensitiveSequenceTagger(cost_matrix=cost_matrix, max_itr=1000, solver='gurobi', max_update=200000,
            reg_constant=reg_constant, learning_rate=learning_rate)

    # ast, best_param = grid_search(ast, X_tr, y_seq, val_idx)
    # print ("best parameter: " + str (best_param) )

    ast.fit(X_tr, y_seq)

    # save for plotting
    np.savetxt('training_objectives.txt', np.column_stack((ast.epoch_times, ast.average_objective)), delimiter=',')

    # predict
    total_cost, total_length = 0.0, 0
    y_pred = [y+1 for y in ast.predict(X_ts)]
    with open('predictions.txt', 'wt') as f:
        for yp in y_pred:
            f.write(",".join([str(i) for i in yp]) + "\n")
    with open('prediction_result.txt', 'wt') as f:
        for y, yp in zip(y_ts, y_pred):
            cost = cost_matrix[yp-1, y-1].sum()
            total_cost += cost
            total_length += len(y)
            f.write("%d,%f,%f\n"%(len(y), cost, cost/len(y)) )

    print ('micro average loss:', total_cost / total_length)
            
    

if "__main__" == __name__:
    Usage = "PYTHONPATH=<Adv lib parent> python <this file> <data dir> <cost file>"
    if len(sys.argv) < 3:
        print ("Usage:", Usage) 
        exit(0)

    data_dir = os.path.abspath(sys.argv[1])
    cost_file = os.path.abspath(sys.argv[2])
    svm_data_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
    run(data_dir, cost_file, svm_data_dir)
    