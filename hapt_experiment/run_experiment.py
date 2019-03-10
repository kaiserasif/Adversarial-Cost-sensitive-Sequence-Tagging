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

import os, sys, ast
import pickle

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

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
    
    # X_tr = [X_tr[seq[1]:seq[1]+5] for seq in seq_tr] # 0 based and end-inclusive split info
    # y_tr = [y_tr[seq[1]:seq[1]+5] for seq in seq_tr]

    X_ts = [X_ts[seq[1]:seq[2]+1] for seq in seq_ts]
    y_ts = [y_ts[seq[1]:seq[2]+1] for seq in seq_ts]

    return X_tr, y_tr, X_ts, y_ts


def preprocess(X_tr, X_ts, poly_degree=1):
    """
    If current directory contains RBFSampler.txt then 
    use RBFSampler, otherwise,
    Do polynomial transform
    also return the combined transform, incase needed

    features are normalized already in the source
    so, only polynomial transformation is done
    default is 1, since 561 fetures is already too many
    """
    rbf_path = os.path.join(os.getcwd(), 'RBFSampler.txt')
    if os.path.exists( rbf_path ):
        with open( rbf_path, 'rt' ) as f:
            kwargs = ast.literal_eval(f.read())
            transformer = RBFSampler(**kwargs)
    else:
        transformer = preprocessing.PolynomialFeatures(degree=poly_degree, interaction_only=False)
    
    X_comb_tr = transformer.fit_transform(np.concatenate(X_tr, axis=0))
    X_comb_ts = transformer.transform(np.concatenate(X_ts, axis=0))

    X_tr = [transformer.transform(x) for x in X_tr]
    X_ts = [transformer.transform(x) for x in X_ts]

    return X_tr, X_ts, X_comb_tr, X_comb_ts, transformer


def grid_search(clf, X_tr, y_seq, val_idx, param_grid):

    # extract the sequences to be used
    X_tr = [X_tr[i] for i in val_idx]
    y_seq = [y_seq[i] for i in val_idx]

    param_grid = {'reg_constant' : (0.0001, 0.001, 0.01)} #, 
                    # 'learning_rate': (0.003, 0.01, 0.03, 0.1, 0.3) } 

    kfold = KFold(3, shuffle=False) # don't shuffle keep consistent splits across algorithms
    gs = GridSearchCV(clf, param_grid, cv=kfold.split(X_tr))
    gs.fit(X_tr, y_seq)
    return gs.best_estimator_, gs.best_params_

class NumpyConcatenator(BaseEstimator):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X, y=None): return np.concatenate(X, axis=0)
    def fit_transform(self, X, y=None): return self.transform(X, y)

class SequenceWrapper(BaseEstimator):
    def __init__(self, transformer, gamma=None, n_components=None, random_state=None): self.transformer = transformer
    def set_params(self, **params): return self.transformer.set_params(**params)
    def get_params(self, deep): 
        params = self.transformer.get_params(deep) 
        params['transformer'] = self.transformer
        return params
    def fit(self, X, y=None): 
        self.transformer.fit(np.concatenate(X, axis=0))
        return self
    def transform(self, X, y=None): return [self.transformer.transform(x) for x in X]
    def fit_transform(self, X, y=None): 
        self.fit(X)
        return self.transform(X)



def grid_search_rbfsampler_advseq(X_tr, y_seq, estimator, val_idx):
    # extract the sequences to be used
    X_tr = [X_tr[i] for i in val_idx]
    y_seq = [y_seq[i] for i in val_idx]
    param_grid = {'adv_seq__reg_constant' : ( 0.0001, 0.001, 0.1), 
                    'rbfsampler__gamma': (100, 200, 500, 1000),
                    'rbfsampler__n_components': (5000, 10000) } 
    kfold = KFold(3, shuffle=False)
    gs = GridSearchCV(estimator, param_grid, cv=kfold.split(X_tr))
    gs.fit(X_tr, y_seq)
    return gs.best_estimator_, gs.best_params_


def run(data_dir, cost_file, svm_data_dir):
    # load data, scale may not be needed
    X_tr, y_tr, X_ts, y_ts = load_hapt_data(data_dir)
    X_tr, X_ts, _, _, transformer = preprocess(X_tr, X_ts)
    print (transformer.__class__.__name__, transformer.get_params())
    
    # load cost_matrix
    cost_matrix = np.loadtxt(cost_file, delimiter=',', dtype=float)

    reg_constant, learning_rate, batch_size = 0.1, 0., 1 # 0 lr for ada_delta update
    # if current dir contains learning parameters, read them
    reg_lr_path = os.path.join(os.getcwd(), 'reg_lr.txt')
    if os.path.exists( reg_lr_path ):
        with open( reg_lr_path, 'rt' ) as f:
            reg_constant, learning_rate, batch_size = ast.literal_eval(f.read())
            print ("reg_constant:", reg_constant, "learning_rate:", learning_rate)

    y_seq = [y-1 for y in y_tr] # for adv_seq, 0 indexed classes
    
    # now create classifier and train
    adv_seq = CostSensitiveSequenceTagger(cost_matrix=cost_matrix, max_itr=1000, solver='gurobi',
            max_update=200000, verbose=3,
            reg_constant=reg_constant, learning_rate=learning_rate, batch_size=batch_size)

    # transformer = RBFSampler(random_state=42)
    # pipe = Pipeline([('rbfsampler', SequenceWrapper(transformer)), ('adv_seq', adv_seq)])
    # val_idx = np.loadtxt(os.path.join(data_dir, 'Train/validation_set.txt'), delimiter=',').astype(int)
    # _, best_params = grid_search_rbfsampler_advseq(X_tr, y_seq, pipe, val_idx)
    # print ("best parameter: " + str (best_params) )
    # transformer.set_params(gamma=best_params['rbfsampler__gamma'], n_components=best_params['rbfsampler__n_components'])
    # adv_seq.set_params(reg_constant=best_params['adv_seq__reg_constant'])
    # transformer.fit(np.concatenate(X_tr, axis=0))
    # X_tr = [transformer.transform(x) for x in X_tr]
    # X_ts = [transformer.transform(x) for x in X_ts]

    # extract some random indices of 20% for grid search
    # val_idx = np.random.permutation(len(X_tr))
    # val_idx = val_idx[:len(X_tr) // 5]

    # save for svm
    if svm_data_dir:
        save_data_svmlight_format(os.path.join(svm_data_dir, 'train.dat'), X_tr, y_tr, start_feature=1)
        save_data_svmlight_format(os.path.join(svm_data_dir, 'test.dat'), X_ts, y_ts, start_feature=1)
        
    # adv_seq, best_param = grid_search(adv_seq, X_tr, y_seq, val_idx)
    # print ("best parameter: " + str (best_param) )

    adv_seq.fit(X_tr, y_seq)

    # save for plotting
    np.savetxt('training_objectives.txt', np.column_stack((adv_seq.epoch_times, adv_seq.average_objective)), delimiter=',')

    # save for future use
    np.savetxt('theta.txt', adv_seq.theta, delimiter=',')
    np.savetxt('transition_theta.txt', adv_seq.transition_theta, delimiter=',')
    
    # save phat and pchecks as merged sequences
    # split them based on sequence lengths
    p_hat, p_check = adv_seq.predict_proba_with_trained_theta(X_ts, cost_matrix, adv_seq.theta, adv_seq.transition_theta)
    np.savetxt('p_hat.txt', np.concatenate(p_hat, axis=0), delimiter=',')
    np.savetxt('p_check.txt', np.concatenate(p_check, axis=0), delimiter=',')
    
    total_cost, total_length = 0.0, 0
    y_pred = [y+1 for y in adv_seq.predict(X_ts)]
    
    # predict
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
    
    # debug training         
    # save phat and pchecks as merged sequences
    # split them based on sequence lengths
    p_hat, p_check = adv_seq.predict_proba_with_trained_theta(X_tr, cost_matrix, adv_seq.theta, adv_seq.transition_theta)
    np.savetxt('p_hat_training.txt', np.concatenate(p_hat, axis=0), delimiter=',')
    np.savetxt('p_check_training.txt', np.concatenate(p_check, axis=0), delimiter=',')
    
    total_cost, total_length = 0.0, 0
    y_pred = [y+1 for y in adv_seq.predict(X_tr)]
    
    # predict
    with open('predictions_training.txt', 'wt') as f:
        for yp in y_pred:
            f.write(",".join([str(i) for i in yp]) + "\n")
    with open('prediction_result_training.txt', 'wt') as f:
        for y, yp in zip(y_tr, y_pred):
            cost = cost_matrix[yp-1, y-1].sum()
            total_cost += cost
            total_length += len(y)
            f.write("%d,%f,%f\n"%(len(y), cost, cost/len(y)) )

    print ('micro average loss (training):', total_cost / total_length)

    save_params = {'transformer' : transformer}
    with open( 'save_params.pkl', 'wb' ) as fid:
        pickle.dump(save_params, fid) 
        
    

if "__main__" == __name__:
    Usage = "PYTHONPATH=<Adv lib parent> python <this file> <data dir> <cost file>"
    if len(sys.argv) < 3:
        print ("Usage:", Usage) 
        exit(0)

    data_dir = os.path.abspath(sys.argv[1])
    cost_file = os.path.abspath(sys.argv[2])
    svm_data_dir = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else None
    run(data_dir, cost_file, svm_data_dir)
    
