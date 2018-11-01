'''
Created on May 7, 2018

@author: kaiser
'''

import numpy as np
import sys
from .zerosum import ZerosumGame
import datetime
import copy

        
class CostSensitiveClassifier():

    def __init__(self, 
                 cost_matrix = None, # if none, generate from y, 0-1 loss
                 max_itr = 100, # epoch of stochastic gradient descent
                 max_update = 10000, # redundant, but controls the update, not epoch
                 game_val_cv = 1.e-3, # coefficient of variation (std/mean) of game values
                 grad_tol = 1e-10, # threshold of gradient of each parameter
                 itr_to_chk = 20, # game values to check for coefficient of variation
                 verbose = 0
                 ):
    
    
        self.cost_matrix = cost_matrix
        self.n_class = None
        if self.cost_matrix is not None:
            self.n_class = self.cost_matrix.shape[1]
        self.max_itr = max_itr
        self.max_update = max_update
        self.game_val_cv = game_val_cv
        self.grad_tol = grad_tol
        self.itr_to_chk = itr_to_chk
        self.verbose = verbose
        
        self.termination_condition = ''
        
    def set_epoch(self, max_itr):
        pass
    
    def fit (self, X, y):
        pass

    def predict (self, X):

        n_feature = X.shape[1] # number of columns
        
        zs = ZerosumGame(self.n_class, self.n_class)
        
        y = np.zeros(X.shape[0])
        
        for i in range(len(y)):
            Cx = self.cost_matrix.copy()
            for c in range(self.n_class): Cx[:, c] += np.dot( X[i], self.theta[c*n_feature:(c+1)*n_feature] )
            # v, p_hat = zs.getRowMinimizerDist(Cx)
            v, p_hat = zs.getRowMinimizerDistLP(Cx)
            
            y[i] = p_hat.argmax()
            
        y = y.astype(int)    
        return self.labels[y]


#### sequence tagging ########
class CostSensitiveSequenceTagger():

    def __init__(self, 
                 cost_matrix = None, # if none, generate from y, 0-1 loss
                 max_itr = 100, # epoch of stochastic gradient descent
                 max_update = 10000, # redundant, but controls the update, not epoch
                 game_val_cv = 1.e-3, # coefficient of variation (std/mean) of game values
                 grad_tol = 1e-10, # threshold of gradient of each parameter
                 itr_to_chk = 20, # game values to check for coefficient of variation
                 verbose = 0
                 ):
    
    
        self.cost_matrix = cost_matrix
        self.n_class = None
        if self.cost_matrix is not None:
            self.n_class = self.cost_matrix.shape[1]
        self.max_itr = max_itr
        self.max_update = max_update
        self.game_val_cv = game_val_cv
        self.grad_tol = grad_tol
        self.itr_to_chk = itr_to_chk
        self.verbose = verbose
        
        self.termination_condition = ''
        
#         self.gp_pchk_lp = gp.Model("p_check_solver")
#         self.gp_pchk_lp.setParam('OutputFlag', 0)
        
    def set_epoch(self, max_itr):
        pass
        
    def solve_pairwise_p_check(self, sequence):
        pass
    
    def compute_feature_expectations(self, x, y, pairwise_pcheck, marginal_pcheck):
        pass

    
    def fit (self, X, Y):
        """
        Trains the model
        Parameters:
        :param X: [sample][state][feature]
        :type X: list of 2-d numpy array
        :param Y: [sample][state]
        :type Y: list of 1-d numpy array
        """
        pass
    
    def predict (self, X):
        Y = []
        # phat using viterbi
        n_class = self.n_class
        n = len(X)
        pair_pot = self.transition_theta 
        theta = self.theta
        for i in range(n): # n samples
            x = X[i]
            T = len(x)
            
            cumu_pot = [[0]*n_class for _ in range(T)]
            history = copy.deepcopy(cumu_pot)
            
            # initial state
            for c in range(n_class):
                cumu_pot[0][c] = np.dot(x[0], theta[:, c])
                
            # rest of the sequence
            for t in range(1, T):
                x_pots = np.dot(x[t], theta)
                for c in range(n_class):
                    hist = 0
                    max_pot = cumu_pot[t-1][hist] + pair_pot[hist][c]
                    for prev_c in range(n_class):
                        prev_pot = cumu_pot[t-1][prev_c] + pair_pot[prev_c][c]
                        if prev_pot > max_pot:
                            max_pot = prev_pot
                            hist = prev_c
                    cumu_pot[t][c] = max_pot + x_pots[c]
                    history[t][c] = hist
                    
            # argmax
            c = np.argmax(cumu_pot[-1])
            y_hat = np.zeros(T)
            y_hat[T-1] = c 
            for t in range(T-1, 0, -1):
                c = history[t][c]
                y_hat[t-1] = c
                
            Y.append(y_hat)
        return Y
        
