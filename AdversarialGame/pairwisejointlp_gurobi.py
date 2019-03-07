import gurobipy as gp
import numpy as np


class PairwiseJointLPSovler:
    """
    Use gurobi 
    for solving pairwise-joint LP for sequence tagging

    """

    def __init__(self, n_class, cost_matrix):
        self.n_class = n_class
        self.cost_matrix = cost_matrix

        self.lp_cache = {}

        print ('gurobi pairwise marginals')


    def _get_lp_for_len(self, T):
        """
        Create a lp for the length T and n_class classes
        """

        if T not in self.lp_cache:

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
            self.lp_cache[T] = (variables, model)

        return self.lp_cache[T]


    def _make_lp(self, sequence, theta, transition_theta):
        """
        Make an LP using the features sequence, cost_matrix, and parameters thetas
        v1 .... -psi(y1, x) -psi(y1,x)... <= 0
        """
        T = len(sequence)
        psi_pairs = transition_theta # there's no other weights. 

        # retrieve a model of the same length
        variables, model = self._get_lp_for_len(T)
        model.reset()

        # based on theta and x
        # update the constraints
        # add first node's constraints
        psi = np.dot(sequence[0,:], theta)
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
            psi = np.dot(sequence[t,:], theta)
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


    def solve_lp(self, sequence, theta, transition_theta):
        """
        Create a pairwise-marginal LP using sequence and parameters
        Solve, and return gamevalue and variable values

        Parameters:
            sequence : ndarray, size T x n_feature 
            theta : ndarray, size n_feature x n_class
            transition_theta : ndarray, size n_class x n_class

        Returns:
            gamevalue : result of maximizing objective
            vars : all variables v and pchecks
        """

        model = self._make_lp(sequence, theta, transition_theta)

        model.optimize()
        if model.Status != gp.GRB.OPTIMAL: print('Gurobi solution status: {}'.format(model.Status) )
        
        vars = np.array([v.x for v in model.getVars()])
        return model.getObjective().getValue(), vars

    
