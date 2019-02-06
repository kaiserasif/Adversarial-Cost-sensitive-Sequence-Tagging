"""
This file includes functions and classes 
to use cvxopt for optimizations
"""

import cvxopt
import numpy as np

class PairwiseJointLPSovler:
    """
    Use cvxopt, http://cvxopt.org/userguide/copyright.html
    for solving pairwise-joint LP for sequence tagging

    minimize c.T @ x # @ = matmult
    s.t. G @ x + slack = h, slack ≥ 0
         A @ x = b
    No, bound parameters, so use the G,h to do so
    """

    def __init__(self, n_class, cost_matrix):
        self.n_class = n_class
        self.cost_matrix = cost_matrix

        self.lp_cache = {}

    def _get_lp_for_len(self, T):
        """
        Create a lp for the length T and n_class classes
        """
        if T not in self.lp_cache:
            n_vars = T + (T-1) * self.n_class ** 2
            # objective max v -> min sum(-v) + 0's for all p's, T-1 joint p's
            objective = cvxopt.matrix([-1.]*T + [0.]*(n_vars - T) )

            # G matrix. make it sparse since only block matrices will have values
            # for p_hat * T rows, we have the constraints
            # v_t ... -\phi -\phi... <= 0
            # then, to set the probability variables' bound, need to set lower bounds
            # -p_y <= 0 # upper bound not needed, equality sum(p) = 1, would bound that
            # total T*y + (n_vars - T) rows, 
            n_rows = T * self.n_class + (n_vars - T)
            G = cvxopt.spmatrix([],[],[], ( n_rows, n_vars ) )
            H = cvxopt.matrix([0.]*n_rows)
            # 1 for T*y v's, then -py for T-1 * y^2 vars
            for t in range(T):
                for yhat in range(self.n_class):
                    G[t*self.n_class + yhat, t] = 1
            for i in range(n_vars-T):
                G[T*self.n_class + i, T + i] = -1

            # A has two sets of matrices.. T-1 probability-simplex constraints, sums to 1
            # then T-2 intermediate nodes' Y equality constraints
            B = cvxopt.matrix( [1.]*(T-1) + [0]*((T-2)*self.n_class) )
            A = cvxopt.matrix( 0., ((T-1) + (T-2) * self.n_class, n_vars) )
            for t in range(T-1):
                A[t, T + t * self.n_class**2 : T + (t+1) * self.n_class**2 ] = 1
                # for \sum_y1 py1y2 = \sum_y3 py2y3
                # for t=0, compute for y(t=1), using t = 0, 1, and 2
                if t < T - 2:
                    var_offset = T + t * self.n_class ** 2 # T-2 pcheck pairse skip
                    for j in range(self.n_class):
                        row = T - 1 + t * self.n_class + j
                        # \sum_i (i, j) = \sum_k (j, k) 
                        # p(i, j) = 1 -> (i * nclass + j) = 1
                        # \sum_k (j, k) = \sum_i (j, i) = (j * n_class + i) = -1
                        for i in range(self.n_class):
                            A[row, var_offset + (i * self.n_class) + j ] = 1
                            A[row, var_offset + self.n_class ** 2 + (j * self.n_class) + i ] = -1

            self.lp_cache[T] = (objective, G, H, A, B)
        
        return self.lp_cache[T]

    def _make_lp(self, x, theta, transition_theta):
        """
        Make an LP using the features x, cost_matrix, and parameters thetas
        """
        T = len(x)
        obj, G, H, A, B = self._get_lp_for_len(T)

        # update the G matrix only
        # based on theta and x

        psi = np.dot(x, theta) # for each time, for each class we have [theta_y * (x)]

        # first node
        # doesn't have pairwise
        for yhat in range (self.n_class):
            for y1 in range (self.n_class):
                for y2 in range (self.n_class):
                    G[yhat, T + y1 * self.n_class + y2] = \
                        self.cost_matrix[yhat, y1] + psi[0, y1]

        # rest of the time-stamps
        for t in range(1, T):
            for yhat in range(self.n_class):
                r = t * self.n_class + yhat
                for yprev in range(self.n_class):
                    for y in range(self.n_class):
                        c = T + (t-1) * self.n_class ** 2 + yprev * self.n_class + y
                        G[r, c] = self.cost_matrix[yhat, y] + \
                            psi[t, y] + transition_theta[yprev, y] # pairwise features are only boolean

        return obj, G, H, A, B

    def solve_lp(self, x, theta, transition_theta):
        obj, G, H, A, B = self._make_lp(x, theta, transition_theta)
        res = cvxopt.solvers.lp(obj, G, H, A, B)
        if res['status'] != 'optimal':
            print(res)
            exit
        return res['primal objective'], res['x']


if "__main__" == __name__:
    pjlp = PairwiseJointLPSovler(2, 1 - np.eye(2))
    for obj in pjlp._get_lp_for_len(4):
        if obj.size[1] > 7:
            for i in range(0, obj.size[1], 6):
                print(obj[:, i:i+6])
        else: print(obj)