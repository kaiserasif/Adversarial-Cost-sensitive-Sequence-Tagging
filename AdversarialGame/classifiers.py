'''
Created on May 7, 2018

@author: kaiser
'''

import numpy as np
from scipy.optimize import minimize, check_grad
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
import time
import copy

from .zerosum import ZerosumGame
from .pairwisejointlp_cvxopt import PairwiseJointLPSovler as Cvxsolver
from .pairwisejoint import PairwiseJoint
from .singleoracle import SingleOracle
from .singleoracle_gurobi import SingleOracle as SingleOracle_g

        
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
                 cost_matrix,
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

        self.solver_object = self.theta = self.transition_theta = None
        self.average_objective = np.zeros(0)
        self.epoch_times = np.zeros(0)
        self.termination_condition = ''


    def set_epoch(self, max_itr):
        self.max_itr = max_itr
    

    def _compute_feature_expectations(self, x, y, pairwise_pcheck, marginal_pcheck):
        """
        Compute feature expectation w.r.t. pcheck values, 
        also compute the sample empirical feature expectation
        Returns:
            pcheck_feat : shape of theta
            empirical_feat : shape of theta
            transition_pcheck_feat : shape of transition theta
            transition_empirical_feat : shape of transition theta
        """
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
        # v, pairwise_pcheck, marginal_pcheck = self.solve_pairwise_p_check(x)
        v, pairwise_pcheck, marginal_pcheck = self.solver_object.solve_for_p_check(
            x, self.theta, self.transition_theta
        )
        pcheck_feat, empirical_feat, transition_pcheck_feat, transition_empirical_feat \
            = self._compute_feature_expectations(x, y, pairwise_pcheck, marginal_pcheck)
        # empirical_expectation = self._compute_empirical_feature_potential(x, Y[i]) # for game value computation
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
        np.random.seed(42)

        n_feature = X[0].shape[1] # number of columns
        n_sample = len(X)
         
        # self.labels = np.unique( np.concatenate(Y) )
        
        # if not self.n_class:
        #     self.n_class = len(self.labels)
             
        # if self.cost_matrix is None:
        #     self.cost_matrix =  1 - np.identity(self.n_class)

        # solver using cvxopt package, could've been in init, 
        # but updated n_class and cost_matrix needed
        # self.cvxsolver = Cvxsolver(self.n_class, self.cost_matrix)
        if "singleoracle" == self.solver:
            self.solver_object = SingleOracle(self.n_class, self.cost_matrix)
        elif "singleoracle_gurobi" == self.solver:
            self.solver_object = SingleOracle_g(self.n_class, self.cost_matrix)
        else:
            self.solver_object = PairwiseJoint(self.n_class, self.cost_matrix, self.solver)
     
        self.theta = np.random.rand(n_feature, self.n_class) - 0.5 
        self.transition_theta = np.random.rand(self.n_class, self.n_class) - 0.5 
             
        avg_objectives = np.zeros(self.max_itr)
        times = np.zeros(self.max_itr)
        start_time = time.process_time()
        perf_time = time.perf_counter()

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
            if update_count >= self.max_update: 
                self.termination_condition = 'max update (%d) exceeded' % update_count
                break
                 
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
            times[itr] = time.process_time() - start_time
            # print(game_val)

            if self.verbose >= 2: 
                print("epoch: %d process_time: %.2f real_time: %.2f average_obj: %.2f" % (itr+1, time.process_time()-start_time, time.perf_counter()-perf_time, avg_objectives[itr]) )
                sys.stdout.flush()
            
            avg_grad = avg_grad / n_sample
            if itr > self.itr_to_chk and  abs(np.std(avg_objectives[itr-self.itr_to_chk:itr+1]) / np.mean(
                avg_objectives[itr-self.itr_to_chk:itr+1])) <= self.game_val_cv:
                self.termination_condition = 'optimization ended: average game value {} after {} iteration'.format(avg_objectives[itr], itr)
                print ('gamevalue termination...')
                break
            if np.all(avg_grad <= self.grad_tol):
                self.termination_condition = 'optimization ended: maximum gradient component {} after {} iteration'.format(avg_grad.max(), itr)
                print ('gradient termination...')
                break

        if len(self.termination_condition) == 0:     
            self.termination_condition = 'Max-iteration ' + str(self.max_itr) +' complete ' + str(avg_objectives[:itr].shape)
        print ('game values: {}'.format(avg_objectives[max(0,itr-10):itr]))
        self.average_objective = avg_objectives[:itr] # per_update_objective[:count] # avg_objectives[:itr]
        self.epoch_times = times[:itr]
        print (self.termination_condition, flush=True)
        


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


    def predict_proba(self, X):
        """
        Use Single Oracle to get the phat distribution
        Parameters:
            X : List of 2-d ndarray
        Returns:
            P_hat : [sample][state, class] list of 2-d ndarray
        """
        return self.predict_proba_with_trained_theta(
            X, self.cost_matrix, self.theta, self.transition_theta)[0]


    def predict_proba_with_trained_theta(self, X, 
        cost_matrix, theta, transition_theta):
        """
        Use Single Oracle to get the phat distribution
        Parameters:
            X : List of 2-d ndarray
            cost_matrix : n_class x n_class 2d ndarray
            theta : feature weights, n_features x n_class 2d ndarray
            transition_theta : edge weights, n_class x n_class 2d ndarray
        Returns:
            P_hat : [sample][state, class] list of 2-d ndarray
            P_check : [sample][state, class] list of 2-d ndarray
        """
        n_class = len(cost_matrix)
        if isinstance(self.solver_object, SingleOracle):
            predictor = self.solver_object
        else:
            predictor = SingleOracle(n_class, cost_matrix)
            
        p_hats = []
        p_checks = []
        for x in X:
            _, phat, _, pcheck = predictor.solve_p_hat_p_check(x, theta, transition_theta)
            p_hats.append( phat.reshape(-1, n_class) )
            p_checks.append ( np.array(pcheck) )

        return p_hats, p_checks


    def score(self, X, Y):
        """
        Compute score
        Scikit-learn expects accuracy measure
        Therefore return -expected_cost
        """
        # Y_pred = self.predict(X)
        # return -self.cost_matrix[np.concatenate(Y_pred), np.concatenate(Y)].mean()  
        p_hats = np.concatenate (self.predict_proba(X), axis=0)
        phatC = np.dot(p_hats, self.cost_matrix.T)
        Y = np.concatenate(Y)
        expected_loss = phatC[np.arange(len(Y)), Y].mean()
        return -expected_loss


    def batch_optimization(self, X, Y):
        start_time = time.process_time()
        perf_time = time.perf_counter()
        self.average_objective, self.epoch_times = [], []
        # make a single 1-D variable of the theta's
        # it will be passed to the fun()
        def fun(theta):
            theta = theta.reshape(-1, self.n_class)
            self.theta = theta[:-self.n_class]
            self.transition_theta = theta[-self.n_class:]
            obj = 0
            for x, y in zip(X, Y):
                v, _, _ = self.solver_object.solve_for_p_check(x, self.theta, self.transition_theta)
                obj += v
                obj -= self._compute_empirical_feature_potential(x, y) 
            obj /= len(Y)
            obj += self.reg_constant * 0.5 * ( sum(sum(self.theta**2)) + sum(sum(self.transition_theta**2)) ) 
            
            return obj

        # gradient function
        def der(theta):
            theta = theta.reshape(-1, self.n_class)
            self.theta = theta[:-self.n_class]
            self.transition_theta = theta[-self.n_class:]
            gradient = np.zeros( self.theta.shape )
            transition_gradient = np.zeros( self.transition_theta.shape )
            for (x, y) in zip(X, Y):
                _, g, tg = self._compute_gradient(x, y)
                gradient += g
                transition_gradient += tg
            gradient /= len(Y)
            transition_gradient /= len(Y)
        
            return np.concatenate((gradient, transition_gradient), axis=0).flatten()

        # for logging
        def callback_fun(theta):
            # global itr
            obj = fun(theta)
            self.average_objective.append(obj)
            self.epoch_times.append(time.process_time() - start_time)
            if self.verbose >= 2:
                print("epoch: %d process_time: %.2f real_time: %.2f objective: %.2f" % (
                    len(self.average_objective), time.process_time()-start_time, time.perf_counter()-perf_time, obj) 
                )
                sys.stdout.flush()
    

        ####
        # now flatten theta and pass
        theta = np.concatenate((self.theta, self.transition_theta), axis=0).flatten()

        res = minimize(fun, theta, method='L-BFGS-B', jac=der, tol=1e-8, callback=callback_fun, options={'maxiter': self.max_itr})
        # print (check_grad(fun, der, res.x))

        ####
        # record the result
        theta = res.x
        theta = theta.reshape(-1, self.n_class)
        self.theta = theta[:-self.n_class]
        self.transition_theta = theta[-self.n_class:]
        # print (res.success, res.status, res.message, res.nit)
    
    

    def _compute_empirical_feature_potential(self, x, y):
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



    def _compute_gradient_from_obj(self, x, y=None, delta=1e-8):
        """
        Compute Objective(Theta)
        Then Theta_i += delta, Compute Objective(delta_i)
        Reset Theta_i -= delta
        Gradient = Objective(delta_i) - Objective(Theta)
        """
        # Compute current objective first
        # obj = self.solve_pairwise_p_check(x, return_objective_only=True)
        obj, _, _ = self.solver_object.solve_for_p_check(x)
        obj += self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )  # reg
        obj -= self._compute_empirical_feature_potential(x, y)

        # compute Objective(Theta + delta)
        tmp_g = np.zeros( self.theta.shape ) 
        tmp_tg = np.zeros( self.transition_theta.shape ) 

        # 1st with theta
        r, c = tmp_g.shape
        for i in range(r):
            for j in range(c):
                self.theta[i, j] += delta # update this theta component
                # v = self.solve_pairwise_p_check(x, return_objective_only=True) # solve for objective
                v, _, _ = self.solver_object.solve_for_p_check(x)
                tmp_g[i, j] = v + self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )
                tmp_g[i, j] -= self._compute_empirical_feature_potential(x, y)
                self.theta[i, j] -= delta # reset this theta component

        # now with transition theta
        r, c = tmp_tg.shape
        for i in range(r):
            for j in range(c):
                self.transition_theta[i, j] += delta # update this theta component
                v, _, _ = self.solver_object.solve_for_p_check(x)
                tmp_tg[i, j] = v + self.reg_constant * 0.5 * ( np.linalg.norm(self.theta)**2 + np.linalg.norm(self.transition_theta)**2 )
                tmp_tg[i, j] -= self._compute_empirical_feature_potential(x, y)
                self.transition_theta[i, j] -= delta # reset this theta component

        # gradient = Objective (Theta + delta in each component) - Objective (Theta)
        # Objective (Theta) is equal for all
        tmp_g -= obj
        tmp_tg -= obj
        return tmp_g/delta, tmp_tg/delta


        def _compute_and_save_gradients(self, X, Y, gradient, transition_gradient):
            """
            For debuggin purpose
            Save passed gradient and computed gradient 
            """
            batch_g = np.zeros(self.theta.shape)
            batch_tg = np.zeros(self.transition_theta.shape)
            for x, y in zip(X, Y):
                tmp_g, tmp_tg = self._compute_gradient_from_obj(x, y, delta=1e-10)
                batch_g += tmp_g
                batch_tg += tmp_tg
            tmp_g = batch_g / len(Y)
            tmp_tg = batch_tg / len(Y)
            tmp = np.column_stack([self.theta.flatten(), tmp_g.flatten(), gradient.flatten(), tmp_g.flatten() - gradient.flatten()])
            np.savetxt("gradient_debug.csv", tmp, delimiter=',')
            tmp = np.column_stack([self.transition_theta.flatten(), tmp_tg.flatten(), transition_gradient.flatten(), tmp_tg.flatten() - transition_gradient.flatten()])
            np.savetxt("transition_gradient_debug.csv", tmp, delimiter=',')
            
