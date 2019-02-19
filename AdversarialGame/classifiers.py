'''
Created on May 7, 2018

@author: kaiser
'''

import numpy as np
from scipy.optimize import minimize, check_grad
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
from .zerosum import ZerosumGame
from .cvxoptsolvers import PairwiseJointLPSovler as Cvxsolver
import gurobipy as gp
import datetime
import copy

        
class CostSensitiveClassifier():

    def __init__(self, 
                 cost_matrix = None, # if none, generate from y, 0-1 loss
                 max_itr = 100, # epoch of stochastic gradient descent
                 max_update = 10000, # redundant, but controls the update, not epoch
                 reg_constant = 1e-2,
                 learning_rate = 0.01,
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
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.game_val_cv = game_val_cv
        self.grad_tol = grad_tol
        self.itr_to_chk = itr_to_chk
        self.verbose = verbose
        
        self.average_objective = []
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
    
        reg_constant = self.reg_constant
    
        # adagrad parameters
        rate = self.learning_rate
        square_g = np.zeros( self.theta.shape ) 
    
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
                self.theta -= rate * gradient / np.sqrt(square_g + 1e-8)

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
            if itr > self.itr_to_chk and  abs(np.std(avg_objectives[itr-self.itr_to_chk:itr+1]) / np.mean(
                avg_objectives[itr-self.itr_to_chk:itr+1])) <= self.game_val_cv:
                self.termination_condition = 'optimization ended: average game value {} after {} iteration'.format(game_val, itr)
                break
            if np.all(avg_grad <= self.grad_tol):
                self.termination_condition = 'optimization ended: maximum gradient component {} after {} iteration'.format(avg_grad.max(), itr)
                break

        if self.termination_condition == '':   
            self.termination_condition = 'Max-iteration ' + str(self.max_itr) +' complete'    
            
        print ('game values: {}'.format(avg_objectives))
        self.average_objective = avg_objectives[:itr]
        
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
class CostSensitiveSequenceTagger(BaseEstimator, ClassifierMixin):

    def __init__(self, 
                 cost_matrix = None, # if none, generate from y, 0-1 loss
                 max_itr = 100, # epoch of stochastic gradient descent
                 max_update = 10000, # redundant, but controls the update, not epoch
                 reg_constant = 1e-2,
                 learning_rate = 0.01, # adagrad. if 0, adadelta
                 game_val_cv = 1.e-3, # coefficient of variation (std/mean) of game values
                 grad_tol = 1e-10, # threshold of gradient of each parameter
                 itr_to_chk = 20, # game values to check for coefficient of variation
                 batch_size = 1, # do a minibatch stochastic update
                 verbose = 1,
                 solver='gurobi' # debugging purpose... compare gplk-cvxopt vs gurobi
                 ):
    
    
        self.cost_matrix = cost_matrix
        self.n_class = 0
        if self.cost_matrix is not None:
            self.n_class = self.cost_matrix.shape[1]
        self.max_itr = max_itr
        self.max_update = max_update
        self.reg_constant = reg_constant
        self.learning_rate = learning_rate
        self.ada_delta_rho = 0.9
        self.game_val_cv = game_val_cv
        self.grad_tol = grad_tol
        self.itr_to_chk = itr_to_chk
        self.batch_size = batch_size
        self.verbose = verbose
        self.solver = solver

        self.average_objective = []
        self.termination_condition = ''
        
        # gurobi model cache for faster build time
        self.gurobimodels = {}

        # cache for linprog models
        self.lineprog_models = {}

    def set_epoch(self, max_itr):
        self.max_itr = max_itr


    def make_linprog_model_of_len(self, T):
        """
        Initialize the objective, and constraint matrices
        Some sub matrices of the Ineq constraint matrix will change
        All other are fixed for len T
        """
        # minimizes the objective 
        # - v1 - v2 ...0 * p1p2 {00, 01, 10, 11} .... 
        # T v variables, T-1 p variable groups, each group with Y^2 variables
        # minimizes only -V
        objective = [-1] * T + [0] * ((self.n_class ** 2) * (T-1))

    
    def get_linprog_model(self, sequence):
        """
        Prepare the objective, A_ub, b_ub, A_eq, b_eq, and bounds
        For the sequence
        """
        pass

    
    def solve_pairwise_p_check_by_linprog(self, sequence):
        """
        Use python's builtin linprog for solving 
        """
        pass
    
        
    def make_gurobi_model_of_len(self, T):

        model = gp.Model("p_check_solver")
        model.setParam('OutputFlag', 0)
        
        # add variables
        variables = []
        for i in range(T):
            variables.append( model.addVar(lb=-gp.GRB.INFINITY, name="v_" + str(i) ) )
        for i in range(1, T): # y1y2, y2y3 : starts from 2nd index
            for a in range(self.n_class):
                for b in range(self.n_class):
                    variables.append( model.addVar(lb=0.0, ub=1.0, name="y_{}_p_({}{})".format(i, a, b) ) )
        # set objective
        model.setObjective(sum(variables[:T]), gp.GRB.MAXIMIZE)

        # add first node's constraints
        for yhat in range(self.n_class):
            lhs = 0
            for a in range(self.n_class):
                for b in range(self.n_class):
                    var_idx = a * self.n_class + b
                    lhs += 0 * variables[T + var_idx] # update this coeff 1
            model.addConstr(lhs >= variables[0], "ct{}y{}".format(0, yhat) )
        # other nodes
        for t in range(1, T):
            # add the inequality, cost matrix
            for yhat in range(self.n_class): # for each yhat add the constraint
                lhs = 0
                for a in range(self.n_class):
                    for b in range(self.n_class):
                        # for 2nd node t=1, y1y2 vars, same as the first node's variables
                        var_idx = (t-1)*(self.n_class**2) + a*self.n_class + b
                        lhs += 0 * variables[T + var_idx] # update this coeff 1
                model.addConstr(lhs >= variables[t], "ct{}y{}".format(t, yhat) )

            # add the probability constraint
            model.addConstr(sum(variables[T + (t-1) * (self.n_class**2) : T + t * (self.n_class**2) ]) == 1, "ceqt{}".format(t))
            # add the pairwise constraints, both previous and next steps needed
            if t < T - 1:
                for b in range(self.n_class):
                    offset = T + (t-1) * (self.n_class**2)
                    lhs = 0
                    # (a=0, b=0), (a=0, b=1), (a=1, b=0), (a=1, b=1)
                    # for a fixed b, add a*n_class + b, for all a
                    for a in range(self.n_class): 
                        lhs += variables[offset + a * self.n_class + b]
                    # rhs, (b=0, c=0), (b=0, c=1), (b=1, c=0), (b=1, c=1)
                    # for fixed b, sum up b*n_class:(b+1)*n_class, move over n_class**2 for the t-1-th step
                    model.addConstr(lhs == sum(variables[offset + (self.n_class**2) + b*self.n_class : offset + (self.n_class**2) + (b+1)*self.n_class]), "cprt{}y{}".format(t, b))

        model.update()
        
        # save the model to the hashmap
        self.gurobimodels[T] = (variables, model)


    def get_gurobi_model(self, sequence):
        ''' get a gurobi model of the same length from hash. update coeff. faster.'''
        
        T = len(sequence)
        psi_pairs = self.transition_theta # there's no other weights. 
        
        # make one if same length lp doesn't exist
        if T not in self.gurobimodels:
            self.make_gurobi_model_of_len(T)

        # retrieve a model of the same length
        variables, model = self.gurobimodels[T]
        model.reset()

        # add first node's constraints
        psi = np.dot(sequence[0,:], self.theta)
        for yhat in range(self.n_class):
            constr = model.getConstrByName("ct{}y{}".format(0, yhat))
            for a in range(self.n_class):
                for b in range(self.n_class):
                    var_idx = a * self.n_class + b
                    model.chgCoeff(constr, variables[T + var_idx], 
                        (self.cost_matrix[yhat, a] + psi[a])
                    )
        # other nodes
        for t in range(1, T):
            psi = np.dot(sequence[t,:], self.theta)
            # add the inequality, cost matrix
            for yhat in range(self.n_class): # for each yhat add the constraint
                constr = model.getConstrByName("ct{}y{}".format(t, yhat))
                for a in range(self.n_class):
                    for b in range(self.n_class):
                        var_idx = (t-1)*(self.n_class**2) + a*self.n_class + b
                        model.chgCoeff(constr, variables[T + var_idx], 
                            ( self.cost_matrix[yhat, b] + psi[b] + psi_pairs[a, b]) 
                        )
        model.update()

        return model
    
    def solve_lp(self, sequence):
        model = self.get_gurobi_model(sequence)

        model.optimize()
        if model.Status != gp.GRB.OPTIMAL: print('Gurobi solution status: {}'.format(model.Status) )
        
        vars = np.array([v.x for v in model.getVars()])
        return model.getObjective().getValue(), vars

    def solve_pairwise_p_check(self, sequence, return_objective_only=False):
        
        T = len(sequence)
        
        if self.solver == 'gurobi':
            obj, vars = self.solve_lp(sequence)
        elif self.solver == 'cvxopt':
            obj, vars = self.cvxsolver.solve_lp(sequence, self.theta, self.transition_theta)
        
        # v = vars[:T]
        vars = vars[T:]
        if return_objective_only: return obj # for lbfgs's objective func, we dont need probabilities

        # # normalize vars, sum is > 1
        # nys = self.n_class ** 2
        # for t in range(T-1):
        #     vars[t * nys : (t+1) * nys] /= vars[t * nys : (t+1) * nys].sum()

        pairwise_pcheck = [ [ [None]*self.n_class for _ in range(self.n_class) ] for _ in range(T-1) ] # T-1 pairs
        marginal_pcheck = [ [0]*self.n_class for _ in range(T) ]
        for i in range(0, T-1): # y1y2, y2y3 : starts from 2nd index
            for a in range(self.n_class):
                for b in range(self.n_class):
                    # var order pipj: (a=0, b=0), (a=0, b=1), (a=1, b=0), (a=1, b=1)
                    p = vars[i*(self.n_class**2) + a*self.n_class + b]
                    pairwise_pcheck[i][a][b] = p
                    marginal_pcheck[i][a] += p
        # final node
        for a in range(self.n_class):
            for b in range(self.n_class):
                # var order pipj: (a=0, b=0), (a=0, b=1), (a=1, b=0), (a=1, b=1)
                p = vars[(T-2)*(self.n_class**2) + a*self.n_class + b]
                marginal_pcheck[T-1][b] += p
                
        return obj, pairwise_pcheck, marginal_pcheck
    
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

    def _compute_gradient(self, x, y):
        """
        Compute the stochastic gradient of the single sample
        """
        v, pairwise_pcheck, marginal_pcheck = self.solve_pairwise_p_check(x)
        pcheck_feat, empirical_feat, transition_pcheck_feat, transition_empirical_feat \
            = self.compute_feature_expectations(x, y, pairwise_pcheck, marginal_pcheck)
        # empirical_expectation = self.compute_empirical_feature_potential(x, Y[i]) # for game value computation
        empirical_expectation = (empirical_feat * self.theta).sum() \
            + (transition_empirical_feat * self.transition_theta).sum()

        gradient = pcheck_feat - empirical_feat + self.reg_constant * self.theta
        transition_gradient = transition_pcheck_feat - transition_empirical_feat + self.reg_constant * self.transition_theta
        
        # add regularization and subtract empirical feature expectation
        v += ( 
            self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 ) 
            - empirical_expectation
        )

        return v, gradient, transition_gradient

    
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
        
        if not self.n_class:
            self.n_class = len(self.labels)
             
        if self.cost_matrix is None:
            self.cost_matrix =  1 - np.identity(self.n_class)

        # solver using cvxopt package, could've been in init, 
        # but updated n_class and cost_matrix needed
        self.cvxsolver = Cvxsolver(self.n_class, self.cost_matrix)
     
        self.theta = np.random.rand(n_feature, self.n_class) - 0.5 
        self.transition_theta = np.random.rand(self.n_class, self.n_class) - 0.5 
             
        avg_objectives = np.zeros(self.max_itr)

        # adagrad parameters
        rate = self.learning_rate # 0.5
        square_g = np.zeros( self.theta.shape ) # + 1e-20 # avoid zero division
        square_transition_g = np.zeros( self.transition_theta.shape ) # + 1e-20
        # aditionally for ada-delta
        delta_g = np.zeros( self.theta.shape ) 
        delta_transition_g = np.zeros( self.transition_theta.shape ) 
        eps = 1e-8 # ada-delta is susceptible to eps... weird... too low makes a very low learning rate 
        
        if self.verbose > 0:
            print(n_feature, " features,", n_sample, " samples,", self.max_itr, " epochs")

        self.batch_size = min(self.batch_size, n_sample)
        # batch update and learning rate is 0, then use lbfgs from scipy
        if self.batch_size == n_sample and self.learning_rate == 0:
            self.batch_optimization(X, Y)
            return 
     
        update_count = 0

        for itr in range(self.max_itr):
            if update_count >= self.max_update: break
            if self.verbose >= 2: print("epoch: ", itr); sys.stdout.flush()
                 
            idx = np.random.permutation(n_sample) 
            avg_grad = np.zeros(self.theta.shape)
            batch_game_val = 0
 
            # mini_batch gradient
            cur_batch = 0 # current count, used to determine if batch size met, or last batch which could be smaller
            batch_gradient = np.zeros( self.theta.shape )
            batch_transition_gradient = np.zeros( self.transition_theta.shape )

            for i in idx:

                cur_batch += 1
                x, y = X[i], Y[i]
                
                game_val, gradient, transition_gradient = self._compute_gradient(x, y)

                batch_game_val += game_val
                avg_grad += np.abs(gradient) # abs() due to stochastic. batch shouldn't need it.

                # modifying existing stochastic gradient to minibatch gradient
                if self.batch_size > 1: # otherwise doesn't affect stochastic code
                    batch_gradient += gradient
                    batch_transition_gradient += transition_gradient
                    if cur_batch % self.batch_size == 0 or cur_batch == len(idx): # batch size or last batch
                        actual_batch_size = self.batch_size if cur_batch % self.batch_size == 0 else cur_batch % self.batch_size
                        gradient = batch_gradient / actual_batch_size
                        transition_gradient = batch_transition_gradient / actual_batch_size
                        batch_gradient.fill(0)
                        batch_transition_gradient.fill(0)
                    else:
                        continue # accumulate gradients

                ## ada grad or ada delta
                if (self.learning_rate > 0): # then ada_grad
                    square_g += np.square(gradient)   
                    square_transition_g += np.square(transition_gradient) 
                    # update theta
                    self.theta -= rate * gradient / np.sqrt(square_g + eps)
                    self.transition_theta -= rate * transition_gradient / np.sqrt(square_transition_g + eps)
                else: # if learning_rate is 0, use ada_delta, http://ruder.io/optimizing-gradient-descent/index.html#adadelta
                    square_g = self.ada_delta_rho * square_g + (1. - self.ada_delta_rho) * np.square(gradient)
                    square_transition_g = self.ada_delta_rho * square_transition_g + \
                                (1. - self.ada_delta_rho) * np.square(transition_gradient)
                    cur_delta_g = np.sqrt( delta_g + eps ) / np.sqrt( square_g + eps ) * gradient
                    cur_delta_tr_g = np.sqrt( delta_transition_g + eps ) / np.sqrt( 
                                        square_transition_g + eps ) * transition_gradient
                    delta_g = self.ada_delta_rho * delta_g + (1. - self.ada_delta_rho) * np.square(cur_delta_g)
                    delta_transition_g = self.ada_delta_rho * delta_transition_g + \
                                (1. - self.ada_delta_rho) * np.square(cur_delta_tr_g)
                    # update thetas with cur deltas
                    self.theta -= cur_delta_g
                    self.transition_theta -= cur_delta_tr_g
 
                update_count += 1
                
            # stopping criteria
            # or evaluate expected loss over all samples again here with O(n_sample) time    
            avg_objectives[itr] = batch_game_val / n_sample
            # print(game_val)
            
            avg_grad = avg_grad / n_sample
            if itr > self.itr_to_chk and  abs(np.std(avg_objectives[itr-self.itr_to_chk:itr+1]) / np.mean(
                avg_objectives[itr-self.itr_to_chk:itr+1])) <= self.game_val_cv:
                self.termination_condition = 'optimization ended: average game value {} after {} iteration'.format(game_val, itr)
                break
            if np.all(avg_grad <= self.grad_tol):
                self.termination_condition = 'optimization ended: maximum gradient component {} after {} iteration'.format(avg_grad.max(), itr)
                break

        if self.termination_condition == '':     
            self.termination_condition = 'Max-iteration ' + str(self.max_itr) +' complete ' + str(avg_objectives.shape)
        print ('game values: {}'.format(avg_objectives[max(0,itr-10):itr]))
        self.average_objective = avg_objectives[:itr] # per_update_objective[:count] # avg_objectives[:itr]
        print (self.termination_condition)

        self.gurobimodels.clear()


    def predict (self, X):
        """
        Find Y_hats using Vterbi
        Parameters:
            X : [sample][state][feature] list of 2-D numpy array
        Returns:
            Y: [sample][state] list of 1-D numpy array
        """

        Y = []
        
        n_class = self.n_class
        n = len(X)
        pair_pot = self.transition_theta 
        theta = self.theta

        for i in range(n): # n samples
            x = X[i]
            T = len(x)
            
            # cumu_pot = [[0]*n_class for _ in range(T)]
            cumu_pot = np.zeros((T, n_class))
            history = np.zeros(cumu_pot.shape, dtype=int) 
            
            cumu_pot[0, :] = np.dot(x[0], theta)
                
            # rest of the sequence
            for t in range(1, T):
                x_pots = np.dot(x[t], theta)
                for c in range(n_class):
                    hist = 0
                    max_pot = cumu_pot[t-1, hist] + pair_pot[hist, c]
                    for prev_c in range(n_class):
                        prev_pot = cumu_pot[t-1, prev_c] + pair_pot[prev_c, c]
                        if prev_pot > max_pot:
                            max_pot = prev_pot
                            hist = prev_c
                    cumu_pot[t, c] = max_pot + x_pots[c]
                    history[t, c] = hist
                    
            # argmax
            c = np.argmax(cumu_pot[-1])
            y_hat = np.zeros(T, dtype=int)
            y_hat[T-1] = c 
            for t in range(T-1, 0, -1):
                c = history[t, c] # backtrack target from this step
                y_hat[t-1] = c
                
            Y.append(y_hat)

        return Y


    def score(self, X, Y):
        """
        Compute score
        Scikit-learn expects accuracy measure
        Therefore return -expected_cost
        """
        Y_pred = self.predict(X)
        return -self.cost_matrix[np.concatenate(Y_pred), np.concatenate(Y)].mean()  


    def batch_optimization(self, X, Y):
        # make a single 1-D variable of the theta's
        # it will be passed to the fun()
        def fun(theta):
            theta = theta.reshape(-1, self.n_class)
            self.theta = theta[:-self.n_class]
            self.transition_theta = theta[-self.n_class:]
            obj = 0
            for x, y in zip(X, Y):
                obj += self.solve_pairwise_p_check(x, return_objective_only=True)
                obj -= self.compute_empirical_feature_potential(x, y) 
            obj /= len(Y)
            obj += self.reg_constant * 0.5 * ( sum(sum(self.theta**2)) + sum(sum(self.transition_theta**2)) ) 
            # print (obj)
            return obj

        # gradient function
        def der(theta):
            theta = theta.reshape(-1, self.n_class)
            self.theta = theta[:-self.n_class]
            self.transition_theta = theta[-self.n_class:]
            gradient = np.zeros( self.theta.shape )
            transition_gradient = np.zeros( self.transition_theta.shape )
            for (x, y) in zip(X, Y):
                _, pairwise_pcheck, marginal_pcheck = self.solve_pairwise_p_check(x)
                pcheck_feat, empirical_feat, transition_pcheck_feat, transition_empirical_feat \
                        = self.compute_feature_expectations(x, y, pairwise_pcheck, marginal_pcheck)
                gradient += pcheck_feat - empirical_feat 
                transition_gradient += transition_pcheck_feat - transition_empirical_feat 
            gradient /= len(Y)
            transition_gradient /= len(Y)
            gradient += self.reg_constant * self.theta
            transition_gradient += self.reg_constant * self.transition_theta
            return np.concatenate((gradient, transition_gradient), axis=0).flatten()

        # for logging
        def callback_fun(theta):
            self.average_objective.append(fun(theta))

        ####
        # now flatten theta and pass
        theta = np.concatenate((self.theta, self.transition_theta), axis=0).flatten()

        # check gradient 
        # for _ in range(5):
        #     print (check_grad(fun, der, np.random.random(theta.shape) - 0.5 ))
        # # check probabilities
        # v, pairwise_pcheck, marginal_pcheck = self.solve_pairwise_p_check(X[0])
        # for t in range(len(Y[0])):
        #     pr = [i for sub in pairwise_pcheck[t] for i in sub] if t < len(Y[0]) - 1 else []
        #     print("node:", t, pr, sum(pr), marginal_pcheck[t], sum(marginal_pcheck[t]))

        res = minimize(fun, theta, jac=der, tol=1e-6, callback=callback_fun, options={'maxiter': self.max_itr})
        print (check_grad(fun, der, res.x))

        ####
        # record the result
        theta = res.x
        theta = theta.reshape(-1, self.n_class)
        self.theta = theta[:-self.n_class]
        self.transition_theta = theta[-self.n_class:]
        print (res.success, res.status, res.message, res.nit)
    
    

    def compute_empirical_feature_potential(self, x, y):
        """
        Compute c = theta * feature(x, y)
        Used to correct the objective: Game + P(y) * theta * feature(x, y) - c
        """
        T = len(x)
        c = np.dot(x[0,:], self.theta[:, y[0]])
        for t in range(1, T):
            c += np.dot(x[t,:], self.theta[:, y[t]]) 
            c += self.transition_theta[y[t-1], y[t]]
        return c



    def compute_gradient_from_obj(self, x, y=None, delta=1e-8):
        """
        Compute Objective(Theta)
        Then Theta_i += delta, Compute Objective(delta_i)
        Reset Theta_i -= delta
        Gradient = Objective(delta_i) - Objective(Theta)
        """
        # Compute current objective first
        obj = self.solve_pairwise_p_check(x, return_objective_only=True)
        obj += self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )  # reg
        obj -= self.compute_empirical_feature_potential(x, y)

        # compute Objective(Theta + delta)
        tmp_g = np.zeros( self.theta.shape ) 
        tmp_tg = np.zeros( self.transition_theta.shape ) 

        # 1st with theta
        r, c = tmp_g.shape
        for i in range(r):
            for j in range(c):
                self.theta[i, j] += delta # update this theta component
                v = self.solve_pairwise_p_check(x, return_objective_only=True) # solve for objective
                tmp_g[i, j] = v + self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )
                tmp_g[i, j] -= self.compute_empirical_feature_potential(x, y)
                self.theta[i, j] -= delta # reset this theta component

        # now with transition theta
        r, c = tmp_tg.shape
        for i in range(r):
            for j in range(c):
                self.transition_theta[i, j] += delta # update this theta component
                v = self.solve_pairwise_p_check(x, return_objective_only=True) # solve for objective
                tmp_tg[i, j] = v + self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )
                tmp_tg[i, j] -= self.compute_empirical_feature_potential(x, y)
                self.transition_theta[i, j] -= delta # reset this theta component

        # gradient = Objective (Theta + delta in each component) - Objective (Theta)
        # Objective (Theta) is equal for all
        tmp_g -= obj
        tmp_tg -= obj
        return tmp_g/delta, tmp_tg/delta


        def compute_and_save_gradients(self, X, Y, gradient, transition_gradient):
            """
            For debuggin purpose
            Save passed gradient and computed gradient 
            """
            tmp_g = np.zeros(self.theta.shape)
            tmp_tg = np.zeros(self.transition_theta.shape)
            for x, y in zip(X, Y):
                tmp_g, tmp_tg = self.compute_gradient_from_obj(x, y, delta=1e-10)
                batch_g += tmp_g
                batch_tg += tmp_tg
            tmp_g = batch_g / len(Y)
            tmp_tg = batch_tg / len(Y)
            tmp = np.column_stack([self.theta.flatten(), tmp_g.flatten(), gradient.flatten(), tmp_g.flatten() - gradient.flatten()])
            np.savetxt("gradient_debug.csv", tmp, delimiter=',')
            tmp = np.column_stack([self.transition_theta.flatten(), tmp_tg.flatten(), transition_gradient.flatten(), tmp_tg.flatten() - transition_gradient.flatten()])
            np.savetxt("transition_gradient_debug.csv", tmp, delimiter=',')
            