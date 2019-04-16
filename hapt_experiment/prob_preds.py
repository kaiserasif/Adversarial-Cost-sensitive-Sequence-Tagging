import pandas as pd
import numpy as np

import os, sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler

sys.path.append(os.path.join(sys.path[0], os.pardir))
from AdversarialGame.classifiers import CostSensitiveSequenceTagger
from hapt_experiment.run_experiment import load_hapt_data, preprocess


def main():
    """
    run: PYTHONPATH=<project_dir> python prob_preds.py <data_dir> <out_dir> <cost_file>
    <project_dir> contains AdversarialGame, hapt_experiment/run_experiment 
    <data_dir> contains Train and Test directories
    <out_dir> contains trained model (theta and transition theta)
    <cost_file> contains cost_matrix
    """

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    cost_file = sys.argv[3]


    X_tr, _, X_ts, y_ts = load_hapt_data(data_dir)
    _, X_ts, _, _, _ = preprocess(X_tr, X_ts)

    cost_matrix = np.loadtxt(cost_file, delimiter=',', dtype=float)

    theta = np.loadtxt( os.path.join(out_dir, 'theta.txt'), delimiter=',', dtype=float)
    transition_theta = np.loadtxt(os.path.join(out_dir, 'transistion_theta.txt'), delimiter=',', dtype=float)

    ast = CostSensitiveSequenceTagger(cost_matrix=cost_matrix)
    p, _ = ast.predict_proba_with_trained_theta(X_ts, cost_matrix, theta, transition_theta)
        
    p_hat = np.concatenate(p, axis=0)
    tmp = p_hat.argmax(axis=1) + 1      
    phatC = np.dot(p_hat, cost_matrix.T)

    print ('cost_sensitive: (yhatCy)', cost_matrix[tmp-1, np.concatenate(y_ts)-1].sum())    
    print ('cost_sensitive: (phatCy)', sum( [pc[y-1] for pc,y in zip(phatC, np.concatenate(y_ts))] ))

    np.savetxt(os.path.join(out_dir, 'repredict_p_hat.txt'), p_hat, delimiter=',')


if "__main__" == __name__:
    main()