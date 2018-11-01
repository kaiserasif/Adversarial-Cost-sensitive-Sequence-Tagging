'''
Created on May 7, 2018

@author: kaiser
'''

import numpy as np
import sys
from .zerosum import ZerosumGame
import gurobipy as gp
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
        self.max_itr = max_itr
    
    def fit (self, X, y):

        n_feature = X.shape[1] # number of columns
        n_sample = X.shape[0]
        
        if self.n_class is None:
            self.n_class = len(np.unique(y))
        
        self.labels = np.unique(y)
            
        if self.cost_matrix is None:
            self.cost_matrix =  1 - np.identity(self.n_class)
    
        self.theta = np.zeros((self.n_class * n_feature, 1))
    
        zs = ZerosumGame(self.n_class, self.n_class)
        
        avg_objectives = np.zeros(self.max_itr)
    
        reg_constant = 1e-5
    
        # adagrad parameters
        rate = 0.5
        square_g = np.zeros( self.theta.shape ) + 1e-20 # avoid zero division
    
        if self.verbose > 0:
            print(n_feature, " features,", n_sample, " samples,", self.max_itr, " epochs")
    
        count = 0
        for itr in range(self.max_itr):
            if count > self.max_update: break
            if self.verbose > 0:
                print("epoch: ", itr)
                
            idx = np.random.permutation(n_sample) # range(n_sample) # 
            avg_grad = np.zeros(self.theta.shape)
            game_val = 0

            for i in idx:
                x = X[[i], :]
                Cx = self.cost_matrix.copy()

                for c in range(self.n_class): Cx[:, [c]] += np.dot( x, self.theta[c*n_feature:(c+1)*n_feature] )
                # Cx -= np.dot( x, theta[(Y[i]-1)*n_feature:Y[i]*n_feature] )
                v, p_check = zs.getColumnMaximizerDist(Cx)
        

                gradient =  reg_constant * self.theta # np.zeros(n_feature) if not L2 
                for c in range(self.n_class): 
                    gradient[c*n_feature:(c+1)*n_feature] += p_check[c] * x.T
                gradient[y[i]*n_feature:(y[i]+1)*n_feature] -= x.T
            
                ## ada grad
                square_g += np.square(gradient)    
                self.theta -= rate * gradient / np.sqrt(square_g)

                # termination track
                avg_grad += np.abs(gradient) # abs() due to stochastic. batch shouldn't need it.
                game_val += v
                
                count += 1
                if count > 0 and count % 100 == 0:
                    print("{} updates".format(count))
                if self.max_update and count > self.max_update: break
    
                # print('{} itr {} sample : v: {} p: {} gradient:{} theta:{}'.format(itr, i, v, p_check, gradient.T, theta.T))
                            
            # stopping criteria
            # or evaluate expected loss over all samples again here with O(n_sample) time    
            game_val /= n_sample # average
            avg_objectives[itr] = game_val
            
            avg_grad = avg_grad / n_sample
            if itr > self.itr_to_chk and  abs(np.std(avg_objectives[-self.itr_to_chk:]) / np.mean(
                avg_objectives[-self.itr_to_chk:])) <= self.game_val_cv:
                self.termination_condition = 'optimization ended: average game value {} after {} iteration'.format(game_val, itr)
                break
            if np.all(avg_grad <= self.grad_tol):
                self.termination_condition = 'optimization ended: maximum gradient component {} after {} iteration'.format(avg_grad.max(), itr)
                break

        if self.termination_condition == '':   
            self.termination_condition = 'Max-iteration ' + str(self.max_itr) +' complete'    
            
        print ('game values: {}'.format(avg_objectives))
    
        return self
    
    
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
                 verbose = 1
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
        self.max_itr = max_itr
        
    def solve_pairwise_p_check(self, sequence):
        T = len(sequence)
        
        psi_pairs = self.transition_theta # there's no other weights. 
        
        model = gp.Model("p_check_solver")
        model.setParam('OutputFlag', 0)
        
        # add variables
        variables = []
        for i in range(T):
            variables.append( model.addVar(lb=-gp.GRB.INFINITY, name="v_" + str(i) ) )
        for i in range(1, T): # y1y2, y2y3 : starts from 2nd index
            for a in range(self.n_class):
                for b in range(self.n_class):
                    variables.append( model.addVar(name="y_{}_p_({}{})".format(i, a, b) ) )
        # set objective
        model.setObjective(sum(variables[:T]), gp.GRB.MAXIMIZE)
        # add first node's constraints
        psi = np.dot(sequence[0,:], self.theta)
        for yhat in range(self.n_class):
            lhs = 0
            for a in range(self.n_class):
                for b in range(self.n_class):
                    var_idx = a * self.n_class + b
                    lhs += (self.cost_matrix[yhat, a] +
                            psi[a]
                            ) * variables[T + var_idx]
            model.addConstr(lhs >= variables[0], "ct{}y{}".format(0, yhat) )
        # other nodes
        for t in range(1, T):
            psi = np.dot(sequence[t,:], self.theta)
            # add the inequality, cost matrix
            for yhat in range(self.n_class): # for each yhat add the constraint
                lhs = 0
                for a in range(self.n_class):
                    for b in range(self.n_class):
                        var_idx = (t-1)*(self.n_class**2) + a*self.n_class + b
                        lhs += ( self.cost_matrix[yhat, b] +
                                 psi[b] + psi_pairs[a, b]
                            ) * variables[T + var_idx]
                model.addConstr(lhs >= variables[t], "ct{}y{}".format(t, yhat) )
            # add the probability constraint
            model.addConstr(sum(variables[T + (t-1) * (self.n_class**2) : T + t * (self.n_class**2) ]) == 1, "ceqt{}".format(t))
            # add the pairwise constraints, both previous and next steps needed
            if t < T - 1:
                for b in range(self.n_class):
                    offset = T + (t-1) * (self.n_class**2)
                    lhs = 0
                    for a in range(self.n_class):
                        lhs += variables[offset + a * self.n_class + b]
                    model.addConstr(lhs == sum(variables[offset + (self.n_class**2) + b*self.n_class : offset + (self.n_class**2) + (b+1)*self.n_class]), "cprt{}y{}".format(t, b))
        model.optimize()
    #     print(model.display())
    #     print(model.getObjective().getValue(), model.getVars())
        # return [(v.varName, v.x) for v in model.getVars()] #, model.getObjective().getValue()
        vars = [v.x for v in model.getVars()] #, model.getObjective().getValue()
        v, vars = vars[:T], vars[T:]
        pairwise_pcheck = [ [ [None]*self.n_class for _ in range(self.n_class) ] for _ in range(T-1) ] # T-1 pairs
        marginal_pcheck = [ [0]*self.n_class for _ in range(T) ]
        for i in range(0, T-1): # y1y2, y2y3 : starts from 2nd index
            for a in range(self.n_class):
                for b in range(self.n_class):
                    p = vars[i*(self.n_class**2) + a*self.n_class + b]
                    pairwise_pcheck[i][a][b] = p
                    marginal_pcheck[i][a] += p
        # final node
        for a in range(self.n_class):
            for b in range(self.n_class):
                p = vars[(T-2)*(self.n_class**2) + a*self.n_class + b]
                marginal_pcheck[i][b] += p
                
        return v, pairwise_pcheck, marginal_pcheck
    
    def compute_feature_expectations(self, x, y, pairwise_pcheck, marginal_pcheck):
        T = len(x)
        pcheck_feat = np.zeros(self.theta.shape)
        empirical_feat = np.zeros(self.theta.shape)
        transition_pcheck_feat = np.zeros(self.transition_theta.shape)
        transition_empirical_feat = np.zeros(self.transition_theta.shape)
        
        for t in range(T):
            empirical_feat[:, y[t]] += x[t]
            for a in range(self.n_class):
                pcheck_feat[:, a] += marginal_pcheck[t][a] * x[t]
            if t < T-1:
                transition_empirical_feat[y[t], y[t+1]] += 1
                for a in range(self.n_class):
                    for b in range(self.n_class):
                        transition_pcheck_feat[a, b] += pairwise_pcheck[t][a][b]
        
        return pcheck_feat, empirical_feat, transition_pcheck_feat, transition_empirical_feat

    
    def fit (self, X, Y):
        """
        Trains the model
        Parameters:
        :param X: [sample][state][feature]
        :type X: list of 2-d numpy array
        :param Y: [sample][state]
        :type Y: list of 1-d numpy array
        """
        n_feature = X[0].shape[1] # number of columns
        n_sample = len(X)
         
        self.labels = np.unique( np.concatenate(Y) )
        
        if self.n_class is None:
            self.n_class = len(self.labels)
             
        if self.cost_matrix is None:
            self.cost_matrix =  1 - np.identity(self.n_class)
     
        self.theta = np.zeros((n_feature, self.n_class))
        self.transition_theta = np.zeros((self.n_class, self.n_class))
             
        avg_objectives = np.zeros(self.max_itr)
     
        reg_constant = 1e-2
     
        # adagrad parameters
        rate = 0.5
        square_g = np.zeros( self.theta.shape ) + 1e-20 # avoid zero division
        square_transition_g = np.zeros( self.transition_theta.shape ) + 1e-20
     
        if self.verbose > 0:
            print(n_feature, " features,", n_sample, " samples,", self.max_itr, " epochs")
     
        count = 0
        itr_start_time = datetime.datetime.now()
        n_class = self.n_class
        for itr in range(self.max_itr):
            if count > self.max_update: break
            if self.verbose > 0:
                print("epoch: ", itr)
                 
            idx = np.random.permutation(n_sample) # range(n_sample) # 
            avg_grad = np.zeros(self.theta.shape)
            game_val = 0
 
            seen = 0
            for i in idx:
                x = X[i]
                start_time = datetime.datetime.now()
                v, pairwise_pcheck, marginal_pcheck = self.solve_pairwise_p_check(x)
                pcheck_feat, empirical_feat, transition_pcheck_feat, transition_empirical_feat \
                    = self.compute_feature_expectations(x, Y[i], pairwise_pcheck, marginal_pcheck)
                
                gradient = pcheck_feat - empirical_feat + reg_constant * self.theta
                transition_gradient = transition_pcheck_feat - transition_empirical_feat + reg_constant * self.transition_theta
                
                ## ada grad
                square_g += np.square(gradient)   
                square_transition_g += np.square(transition_gradient) 
                
                self.theta -= rate * gradient / np.sqrt(square_g)
                self.transition_theta -= rate * transition_gradient / np.sqrt(square_transition_g)
 
                game_val += sum(v)

            # stopping criteria
            # or evaluate expected loss over all samples again here with O(n_sample) time    
            game_val /= n_sample # average
            avg_objectives[itr] = game_val
            
            avg_grad = avg_grad / n_sample
            if itr > self.itr_to_chk and  abs(np.std(avg_objectives[-self.itr_to_chk:]) / np.mean(
                avg_objectives[-self.itr_to_chk:])) <= self.game_val_cv:
                self.termination_condition = 'optimization ended: average game value {} after {} iteration'.format(game_val, itr)
                break
            if np.all(avg_grad <= self.grad_tol):
                self.termination_condition = 'optimization ended: maximum gradient component {} after {} iteration'.format(avg_grad.max(), itr)
                break

        if self.termination_condition == '':     
            self.termination_condition = 'Max-iteration ' + str(self.max_itr) +' complete'                 
        print ('game values: {}'.format(avg_objectives[max(0,itr-10):itr]))

    
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
        
# 
#         n_feature = X.shape[1] # number of columns
#         
#         zs = ZerosumGame(self.n_class, self.n_class)
#         
#         y = np.zeros(X.shape[0])
#         
#         for i in range(len(y)):
#             Cx = self.cost_matrix.copy()
#             for c in range(self.n_class): Cx[:, c] += np.dot( X[i], self.theta[c*n_feature:(c+1)*n_feature] )
#             # v, p_hat = zs.getRowMinimizerDist(Cx)
#             v, p_hat = zs.getRowMinimizerDistLP(Cx)
#             
#             y[i] = p_hat.argmax()
#             
#         y = y.astype(int)    
#         return self.labels[y]
        pass
