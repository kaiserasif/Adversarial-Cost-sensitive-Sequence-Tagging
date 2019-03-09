import numpy as np
import gurobipy as gp

class SingleOracle:
    """
    In two player zerosum game,
    Double Oracle and Single Oracle algorithms
    incrementally add choices until no improvement
    is seen from the players' perspective

    Single Oracle algo:
    return P_hat margianls, and P_check pairwise_joints and marginals
    P hats needed in order to find Y_check best response
    start with S = {000, 111, 222 ...} pcheck actions of the seq lengths
    minimize v + 0.(p_hat (TxY))
    such that v >= theta * (x, y_check) + sum_T p_hat_T * C_T[:y_check], for y_check in S
    i.e. for each of the actions in S, add a inequality constraint
    equality is for each timestamp t in T, sum p_hat = 1
    after getting P_hat, find best response using V_terbi
    if best_response value is smaller than the game value while solving P_hat, break
    add best response and repeat
    after break, solve for P_check, P_hat should be unchanged
    compute marginal and pariwise joints from the P_check combinations
    return
    In java implementation, after P_hat computation in each step, 
    constraints are checked if they same similar value
    if so, then corresponding p_hats are uniformly distributed
    probably that's a sanity check, skipped here
    """

    def __init__(self, n_class, cost_matrix, max_itr = 10000):
        """
        Initialize SingleOracle with the references to the 
        Theta parameters learned in the optimization.
        They are used in finding the best reponses.
        Parameters:
        -----------
            n_class : number of target classes
            cost_matrix : numpy matrix of n_class x n_class
            max_itr : maximum iteration for finding best reponses
        """
        self.n_class = n_class
        self.cost_matrix = 1.0 * cost_matrix # int array causes problem with cvxopt matrix data conversion
        self.max_itr = max_itr

        # could be too many, 
        # but gurobi is slow in creating new ones
        # trade off
        self.phat_lp_cache = {} 
        self.pchek_lp_cache = {}

        print ('gurobi single oracle')


    def set_param(self, **kwargs):
        """
        this method provides the ability to set the updated 
        parameters
        """
        self.__dict__.update(kwargs)


    def solve_p_hat_p_check(self, sequence, theta, transition_theta):
        """
        The Single Oracle method

        Parameters:
        -----------
            sequence: the X or feature values of the sequence
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        Returns:
        --------
            v : Number : game value
            p_hat : 1d-numpy array of length T x Y
            p_check joints : numpy array of size T x Y x Y
            p_check marginals : numpy array of size T x Y 
        """

        T = len(sequence)

        # initialize actions with all same class for full sequence for all classes
        pcheck_actions = []
        for c in range(self.n_class): pcheck_actions.append( [c] * T ) 
        
        # construct the lp and solve for P_hat
        """ Minimize: c^T * x
                Subject to: A_ub * x >= b_ub
                A_eq * x == b_eq 
        """
        # get the model without the y_check_action contraints
        model = self._initialize_singleoracle_phat_lp(T)
        for ia in range(len(pcheck_actions)): 
            self._add_singleoracle_constraint_for_pcheck_action(sequence, model,
                pcheck_actions, ia, theta, transition_theta)
            
        # use cvxopt for sparse matrices?
        # http://cvxopt.org/userguide/coneprog.html#linear-programming 
        # but sparse/dense matrices in cvxopt cannot be changed in size
        # sparse matrix cannot shrink by making some entries 0

        # start the loop of solving and finding the best response
        previous_min_gamevalue = None
        loop_count, reached_stop = 0, False
        while (loop_count < self.max_itr and not reached_stop):
            loop_count += 1

            # solve the lp
            val, vars_ = self._solve_phat_lp(model)
            
            # redistribute deterministic phats
            # at the corners, i.e. equal contraints, 
            # phat can take any of the possible determinitically,
            # which throws off the pcheck-best-response
            vars_ = self._adjust_phat(val, vars_,
                        sequence, pcheck_actions, theta, transition_theta)
            
            phat = vars_[1:] # variables are: v, phat11, phat12, phat21,..., phatTY
            min_gamevalue = val # v is the game value / objective value, since all other objective coeff are 0


            if min_gamevalue == previous_min_gamevalue:
                reached_stop = True
                break
            previous_min_gamevalue = min_gamevalue

            # call best response function
            new_action_val, new_action = self._find_singleoracle_best_pcheck(sequence,
                phat, theta, transition_theta)
            
            # if value is less than the p-hat solution, break
            if new_action_val <= min_gamevalue: break # adversary didn't find a choice to maximize cost

            # add best response to the actions, repeat
            if new_action not in pcheck_actions :
                pcheck_actions.append(new_action)
                self._add_singleoracle_constraint_for_pcheck_action(sequence, model,
                pcheck_actions, len(pcheck_actions)-1, theta, transition_theta)
        
        # after breaking the loop, call for pcheck solution
        max_gamevalue, pcheck_dist = self._pcheck_singleoracle(sequence,  pcheck_actions, theta, transition_theta)
        # debug 
        # if max_gamevalue != previous_min_gamevalue:
        #     print ("max vs min games: ", max_gamevalue, previous_min_gamevalue, max_gamevalue - previous_min_gamevalue)
        
        # the pcheck is for the ycheck actions, 
        # compute marginals and pairwise-joints 
        pairwise_pcheck, marginal_pcheck = self._compute_marginal_and_pairwise(pcheck_dist, pcheck_actions)
        
        return previous_min_gamevalue, phat, pairwise_pcheck, marginal_pcheck


    def solve_for_p_check(self, sequence, theta, transition_theta):
        """
        The Single Oracle method

        Parameters:
        -----------
            sequence: the X or feature values of the sequence
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        
        Returns:
        --------
            v : Number : game value
            p_check joints : numpy array of size T x Y x Y
            p_check marginals : numpy array of size T x Y
        """
        gamevalue, phat, pairwise_pcheck, marginal_pcheck = self.solve_p_hat_p_check(
            sequence, theta, transition_theta)
        return gamevalue, pairwise_pcheck, marginal_pcheck


    def _initialize_singleoracle_phat_lp(self, T):
        """
        phat single oracle lp is:
        min v + (phat 0's), such that phat >= 0, sum phat_t = 1 for each t
        and v >= potential for each ycheck combination + sum phat * cost of ycheck combination
        Parameters:
            T : length of the sequence
        Returns:
            gurobi model
        """
        if T not in self.phat_lp_cache:
            n_class = self.n_class

            model = gp.Model("single_oracle_p_hat_solver")
            model.setParam('OutputFlag', 0)

            variables = []
            # add the v variable, not bounded
            variables.append( model.addVar(lb=-gp.GRB.INFINITY, name="v" ) )
            # add the p_hat variables
            for t in range(T):
                for y in range(n_class):
                    variables.append( model.addVar(lb=0.0, ub=1.0, name="y_%d_p_%d"%(t, y) ) )
            # set objective
            model.setObjective(variables[0], gp.GRB.MINIMIZE)
            
            # add equality constraints
            for t in range(T):
                phat_index = 1 + t * n_class
                # for each timestamp t,
                # 1 (v offset) + t * nclass prev vars, 
                # to 1 + (t + 1) * n_class p_hats sums to 1
                model.addConstr(sum(variables[phat_index : phat_index + n_class]) == 1, "ceqt_%d"%(t))

            model.update()
            self.phat_lp_cache[T] = model
        
        return self.phat_lp_cache[T].copy()


    def _add_singleoracle_constraint_for_pcheck_action(self, sequence, model,
        pcheck_actions, ia, theta, transition_theta):
        """
        compute A_ub >= b_ub constraint for the specified pheck action
        v >= potential + phat * cost (for all T)
        Parameters:
            sequence : numpy.2d-array
            model : gurobi model where the constraint will be added
            pcheck_action : list [ numpy.1d-array ]
            ia : index of action to add
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        """
        action = pcheck_actions[ia]
        T = len(action)
        n_class = self.n_class
        potential = 0.
        lhs = model.getVarByName("v")
        rhs = 0.

        for t in range(T):
            # \sum cost[phat_t, p_check_t] 
            for y in range(n_class):
                # for t-th node, cost is phat's row vs p-check's t-th action
                rhs += model.getVarByName("y_%d_p_%d"%(t, y)) * self.cost_matrix[y, action[t]]

            # compute the theta * features
            # similar to empirical features, since not stochastic
            potential += np.dot(sequence[t,:], theta[:, action[t]])
            if t > 0:
                potential += transition_theta[action[t-1], action[t]]

        rhs += potential
        model.addConstr(lhs >= rhs, "c_action_%d"%ia)

        model.update()


    def _solve_phat_lp(self, model):
        """
        use cvxopt.solvers.lp with glpk to solve the lp
        Parameters:
            model : gurobi model
        Returns:
            primal_objective : float
            variables : np.array 1d
        """
        model.reset()
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL: print('Gurobi solution status: {}'.format(model.Status) )

        vars_ = np.array([v.x for v in model.getVars()])
        return model.getObjective().getValue(), vars_


    
    def _adjust_phat(self, val, vars_,
                        sequence, pcheck_actions, theta, transition_theta):
        """
        redistribute deterministic phats
        at the corners, i.e. equal contraints, 
        phat can take any of the possible determinitically,
        which throws off the pcheck-best-response.
        Uses _find_singleoracle_best_pcheck()
        Parameters:
            val : gamevalue, wasn't needed, vars_[0] should suffice
            vars_ : v + phats
            sequence : the features of the sequence
            pcheck_actions : the p_check actions
            theta : feature weights
            transition_theta: edge weights
        Returns:
            vars_ : adjusted phat values
        """
        # print ('checking equal constraints...')
        epsilon = 1e-8
        T = len(pcheck_actions[0])
        n_class = self.n_class
        # first check how many constraints are equal 
        # A_ub_list also contains the phat >= 0 conditions at the beginning
        eqaulconstraints = 0

        for ia in range(len(pcheck_actions)):
            potential = 0.
            lhs = vars_[0]
            rhs = 0.
            action = pcheck_actions[ia]

            for t in range(T):
                # \sum cost[phat_t, p_check_t] 
                for y in range(n_class):
                    # for t-th node, cost is phat's row vs p-check's t-th action
                    rhs += vars_[1 + t * n_class + y] * self.cost_matrix[y, action[t]]

                # compute the theta * features
                # similar to empirical features, since not stochastic
                potential += np.dot(sequence[t,:], theta[:, action[t]])
                if t > 0:
                    potential += transition_theta[action[t-1], action[t]]

            rhs += potential

            # check if constraint was met with equality
            if abs ( lhs - rhs ) < epsilon: eqaulconstraints += 1

        # if none, nothing to adjust
        if eqaulconstraints <= 1: return vars_
        
        # debug:
        # print ('equal constraints:', eqaulconstraints)

        # for each set of T phats, 
        # adjust if multiple yhat could acheive same 
        # constraint values based on pcheck_action_distribution

        max_gamevalue, pcheck_dist = self._pcheck_singleoracle(sequence,  pcheck_actions, theta, transition_theta)
        if abs(max_gamevalue - vars_[0]) < epsilon: print ("_adjust_phat() game values:", max_gamevalue, vars_[0])

        det_phats = abs( vars_[1:] - 1 ) < epsilon
        for t in range(T):
            if det_phats[t * n_class : (t+1) * n_class].any():
                det_idx = -1
                for y_hat in range(n_class):
                    if det_phats[t * n_class + y_hat]:
                        det_idx = y_hat
                        break
                # print ('deterministic phats: t=%d, y=%d'%(t, det_idx) )
                # otherwise nothing to adjust for this time step
                # compute the Cpcheck
                Cpcheck = [0] * n_class # for Y phats
                for ia, pcheck_a in enumerate(pcheck_dist):
                    for y_hat in range(n_class):
                        Cpcheck[y_hat] += self.cost_matrix[y_hat, pcheck_actions[ia][t]] * pcheck_a

                # count equal 
                equal_y_hats = [abs(Cpcheck[det_idx] - Cpcheck[y]) < epsilon for y in range(n_class) ]
                count = sum( equal_y_hats )
                if count > 1:
                    for y_hat in range(n_class):
                        if equal_y_hats[y_hat]: vars_[1 + t * n_class + y_hat ] = 1.0 / count
                    print ('equal p_hats: %d' % count, Cpcheck)
        return vars_

        

    def _find_singleoracle_best_pcheck(self, sequence, phat, theta, transition_theta):
        """
        find the best y_check action that maximizes the loss
        Parameters:
            sequence : numpy 2-d array
            phat : numpy 1-d array of size of T * n_class
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        Returns:
            List[int] : T length y_check action with maximum loss
        """
        # max_ycheck ( E_phat[cost[ycheck]] + feature_potential(ycheck) )
        # max_y1 ( E_phat(1)[cost[y1]] + pot(y1, x1) 
        #   + max_y2 ( pot(y1, y2) + E_phat(2)[cost[y2]] + pot(y2, x2) 
        #       + max...) )
        # viterbi, similar to potential based prediction, 
        # with the addition of expected cost
        # above goes from end to front, we do front to back
        T = len(sequence)
        x = sequence

        pair_pot = transition_theta # links are only boolean

        cumu_pot = np.zeros((T, self.n_class))
        history = np.zeros(cumu_pot.shape, dtype=int) 

        cumu_pot[0, :] = ( np.dot(x[0], theta) # local feature feature potential    
                        + np.dot(phat[:self.n_class], self.cost_matrix) # expectation: phat * cost
        )
                
        # rest of the sequence
        for t in range(1, T):
            x_pots_costs = ( np.dot(x[t], theta)
                    + np.dot(phat[t * self.n_class : (t+1) * self.n_class], self.cost_matrix)
            )
            for y_check in range(self.n_class):
                hist = 0
                max_pot = cumu_pot[t-1, hist] + pair_pot[hist, y_check]
                for prev_y in range(self.n_class):
                    prev_pot = cumu_pot[t-1, prev_y] + pair_pot[prev_y, y_check]
                    if prev_pot > max_pot:
                        max_pot = prev_pot
                        hist = prev_y
                cumu_pot[t, y_check] = max_pot + x_pots_costs[y_check]
                history[t, y_check] = hist
                
        # argmax
        c = np.argmax(cumu_pot[-1])
        y_check = [-1] * T
        y_check[T-1] = c 
        for t in range(T-1, 0, -1):
            c = history[t, c] # backtrack target from this step
            y_check[t-1] = c

        return np.max(cumu_pot[-1]), y_check


    def _pcheck_singleoracle(self, sequence, pcheck_actions, theta, transition_theta):
        """
        Compute pcheck distribution using single oracle method
        Parameters :
            sequence : 2d np.array
            pcheck_actions : List[List[float]]
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        Returns : 
            gamevalue : float
            pcheck : List[float] of len(pcheck_actions)
        """
        # max_{pchecks, v_t for t} sum v_t + potential (p_check)
        # s.t. sum pchecks = 1 # A_eq = b_eq, one constraint
        # v_t <= C[yhat(t)] * pchecks, for all t, for all y_hat
        # cannot be cached, since set of actions may vary lp to lp
        # constraints and objective both depends on it, and too many combinations

        x = sequence
        n_action = len(pcheck_actions)
        T = len(sequence)
        n_class = self.n_class

        model = gp.Model("single_oracle_p_check_solver")
        model.setParam('OutputFlag', 0)
            
        # add variables
        variables = []
        for t in range(T):
            variables.append( model.addVar(lb=-gp.GRB.INFINITY, name="v_%d"%t ) )
        for ia in range(n_action): # one var for each pcheck action
            variables.append( model.addVar(lb=0.0, ub=1.0, name="y_action_%d"%ia ) )
            
        objective = [1.] * (T) + [0.] * n_action
        
        for t in range(T):
            for i in range(n_action):
                objective[T + i] += np.dot(x[t], theta[:, pcheck_actions[i][t]])
                if t > 0: # pairwise links
                    objective[T + i] += transition_theta[pcheck_actions[i][t-1], pcheck_actions[i][t]]
            
            # v_t <= C[yhat(t)] * pchecks, for all t, for all y_hat
            for yhat in range(n_class):
                # v_t for each phat
                rhs = 0.
                # - C[yhat(t), ychecks[t]] for each ycheck actions, after T variables
                for i in range(n_action):
                    if self.cost_matrix[yhat, pcheck_actions[i][t]]:
                        rhs += self.cost_matrix[yhat, pcheck_actions[i][t]]

                model.addConstr(variables[t], gp.GRB.LESS_EQUAL, rhs, "c_y_%d_p_%d" % (t, yhat) )

        # set the probability simplex constraint
        model.addConstr(sum(variables[T: ])==1, "c_prob")

        # set the objective
        model.setObjective(
            sum([v * o for v,o in zip(variables, objective)]),
            gp.GRB.MAXIMIZE
        )

        model.update()

        # solve it
        model.optimize()

        if model.Status != gp.GRB.OPTIMAL: print('Gurobi non-optimal solution status: %d'%model.Status )
        vars_ = np.array([v.x for v in model.getVars()])
        return model.getObjective().getValue(), vars_[T:]


    def _compute_marginal_and_pairwise(self, pcheck_dist, pcheck_action):
        """
        from pcheck actions' probabilities
        compute the pairwise and marginal pchecks for each T
        Parameters:
            pcheck_dist : 1d ndarray of len(pcheck_action)
            pcheck_action : List[List[float]]
        Returns:
            pairwise_pcheck : List[List[List[float]]] of y1y2, y2y3... size: T-1 x n_class x n_class
            marginal_pcheck : List[List[float]] of y1, y2, y3... size T x n_class
        """
        n_action, T = len(pcheck_action), len(pcheck_action[0])

        pairwise_pcheck = [ [ [0]*self.n_class for _ in range(self.n_class) ] for _ in range(T-1) ] # T-1 pairs
        marginal_pcheck = [ [0]*self.n_class for _ in range(T) ]

        for i in range(n_action):
            for t in range(T):
                marginal_pcheck[t][ pcheck_action[i][t] ] += pcheck_dist[i]
                if t < T - 1:
                    # y1y2, y2y3 : starts from 2nd index
                    pairwise_pcheck[t][ pcheck_action[i][t] ] [pcheck_action[i][t+1]] += pcheck_dist[i]
                
        return pairwise_pcheck, marginal_pcheck


if "__main__" == __name__:
    n_class = 3
    T = 10
    n_feature = 15
    theta = np.random.randint(-5, 5, (n_feature, n_class))
    transition_theta = np.random.randint(-5, 5, (n_class, n_class))
    x = np.random.randint(-5, 5, (T, n_feature))
    # print(theta, transition_theta, x)

    so = SingleOracle(n_class, 1-np.eye(n_class))
    print ( so.solve_p_hat_p_check(x*1., theta*1.0, transition_theta*1.0) )
