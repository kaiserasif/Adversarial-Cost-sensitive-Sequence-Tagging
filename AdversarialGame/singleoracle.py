import numpy as np
# from scipy.optimize import linprog # cannot rely on linprog
import cvxopt

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

        print ('cvxopt single oracle')


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
                Subject to: A_ub * x <= b_ub
                A_eq * x == b_eq 
        """
        objective, A_ub_list, b_ub_list, A_eq, b_eq = self._initialize_singleoracle_phat_lp(T)
        for action in pcheck_actions: 
            self._add_singleoracle_constraint_for_pcheck_action(sequence, A_ub_list, b_ub_list,
                action, theta, transition_theta)
            
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
            val, vars_ = self._solve_phat_lp(objective, A_ub_list, b_ub_list, A_eq, b_eq)
            
            # redistribute deterministic phats
            # at the corners, i.e. equal contraints, 
            # phat can take any of the possible determinitically,
            # which throws off the pcheck-best-response
            vars_ = self._adjust_phat(val, vars_, A_ub_list, b_ub_list,
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
                self._add_singleoracle_constraint_for_pcheck_action(sequence, A_ub_list, b_ub_list,
                    new_action, theta, transition_theta)
        
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
        min v + (phat 0's), such that phat >= 0, sum phat = 1 for each t
        and v >= potential for each ycheck combination + sum phat * cost of ycheck combination
        i.e. -v + phat * cost (for all T) <= -potential ; for all ycheck combination
        cvxopt don't have bounds, so phat >= 0 variables should also be in A_ub <= b_ub matrix
        for var = 2 to T+1, -var <= 0 in A_ub <= b_ub matrix
        Parameters:
            T : length of the sequence
        Returns:
            objective : 1 + 0 + .... + 0 (1 + n_class * T length)
            A_ub_list : no pcheck actions added, only -phat <= 0
            b_ub_list : 0 for each phat, lists for enabling appending new actions
            A_eq : 1+...+1 for each set of the T phats 
            b_eq : 1 for each set of T phats
        """
        n_phats = T * self.n_class
        
        objective = cvxopt.matrix([1.] + [0.] * n_phats )
        
        A_ub_list = [[0] * (1 + n_phats) for _ in range(n_phats)]
        A_eq = cvxopt.spmatrix([],[],[], ( T, 1 + n_phats ) )
        for t in range(T):
            for c in range(self.n_class):
                phat_index = t * self.n_class + c
                # column offset 1 for the v variable, which is unbounded
                # for each of the phat vars, -1p <= 0
                A_ub_list[phat_index][1 + phat_index] = -1

                # for each timestamp t, one row
                # in each row, 1 (v offset) + t * nclass prev vars, 
                # then nclass c vars 1 
                # sums to 1 (b_eq)
                A_eq[t, 1 + phat_index] = 1

        b_ub_list = [0] * n_phats
        b_eq = cvxopt.matrix(1., ( T, 1 ) )
        
        return objective, A_ub_list, b_ub_list, A_eq, b_eq


    def _add_singleoracle_constraint_for_pcheck_action(self, sequence, A_ub_list, b_ub_list,
        action, theta, transition_theta):
        """
        compute A_ub <= b_ub constraint for the specified pheck action
        - v + phat * cost (for all T) <= - potential
        Parameters:
            sequence : numpy.2d-array
            A_ub_list : List[List[float]]
            b_ub_list : List[float]
            action : numpy.1d-array
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        """
        T = len(action)
        n_phats = T * self.n_class
        new_row = [-1.] + [0.] * n_phats
        potential = 0.

        for t in range(T):

            # -v + cost[phat_t, p_check_t]..
            for c in range(self.n_class):
                # for t-th node, cost is phat's row vs p-check's t-th action
                new_row [1 + t * self.n_class + c] = self.cost_matrix[c, action[t]]

            # compute the theta * features
            # similar to empirical features, since not stochastic
            potential += np.dot(sequence[t,:], theta[:, action[t]])
            if t > 0:
                potential += transition_theta[action[t-1], action[t]]

        # append this new constraint to A_ub and b_ub
        A_ub_list.append(new_row)
        b_ub_list.append(-potential)


    def _solve_phat_lp(self, objective, A_ub_list, b_ub_list, A_eq, b_eq):
        """
        use cvxopt.solvers.lp with glpk to solve the lp
        Parameters:
            objective : 1-d cvxopt dense matrix
            A_ub_list : List[List[float]]
            b_ub_list : List[float]
            A_eq : cvxopt.spmatrix
            b_eq : 1-d cvxopt.matrix
        Returns:
            primal_objective : float
            variables : np.array 1d
        """
        # should this be sparse? 
        # another conversion will be needed
        # but solving lp, does it give low performance with 
        # dense matrix having -phat <= 0 for T * nclass rows?
        # cvxopt converts a list in column major order
        # so np.array is created to preserve the orientation
        A_ub = cvxopt.matrix (np.array(A_ub_list)) 
        b_ub = cvxopt.matrix (b_ub_list)
        # diable glpk messages
        # cvxopt.solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_OFF'}
        res = cvxopt.solvers.lp(objective, A_ub, b_ub, A_eq, b_eq, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        if res['status'] != 'optimal':
            print(res)
            exit
        return res['primal objective'], np.array(res['x'])[:, 0] # a column vector of dense matrix, convert to 1d

    
    def _adjust_phat(self, val, vars_, A_ub_list, b_ub_list,
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
            A_ub_list : constraints lhs
            b_ub_list : constraints rhs
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
        n_phats = T * n_class # skip phat prob contraints
        eqaulconstraints = 0
        for ia in range(n_phats, len(b_ub_list)):
            # check if constraint was met with equality
            if abs ( vars_[0] + sum([A_ub_list[ia][iv] * vars_[iv] 
                        for iv in range(1, n_phats + 1)])
                         - b_ub_list[ia]
            ) < epsilon: eqaulconstraints += 1

        # if none, nothing to adjust
        if eqaulconstraints <= 1: return vars_
        
        # debug:
        # print ('equal constraints: ', eqaulconstraints)

        # for each set of T phats, 
        # adjust if multiple yhat could acheive same 
        # constraint values based on pcheck_action_distribution

        max_gamevalue, pcheck_dist = self._pcheck_singleoracle(sequence,  pcheck_actions, theta, transition_theta)
        if abs(max_gamevalue - vars_[0]) < epsilon: print ("_adjust_phat():", max_gamevalue, vars_[0])

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
                        Cpcheck[y_hat] += A_ub_list[n_phats + ia][1 + T * n_class + y_hat] * pcheck_a

                # count equal 
                equal_y_hats = [abs(Cpcheck[det_idx] - Cpcheck[y]) < epsilon for y in range(n_class) ]
                count = sum( equal_y_hats )
                if count > 1:
                    for y_hat in range(n_class):
                        if equal_y_hats[y_hat]: vars_[1 + t * n_class + y_hat ] = 1.0 / count
                    print ('equal p_hats: %d' % count)
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
        tolerance = 1e-6

        pair_pot = transition_theta # links are only boolean

        cumu_pot = np.zeros((T, self.n_class))
        history = np.zeros(cumu_pot.shape, dtype=int) 
        multi_history = {}

        cumu_pot[0, :] = ( np.dot(x[0], theta) # local feature feature potential    
                        + np.dot(phat[:self.n_class], self.cost_matrix) # expectation: phat * cost
        )
                
        # rest of the sequence
        for t in range(1, T):
            if t not in multi_history:
                multi_history[t] = {}
            x_pots_costs = ( np.dot(x[t], theta)
                    + np.dot(phat[t * self.n_class : (t+1) * self.n_class], self.cost_matrix)
            )
            for y_check in range(self.n_class):
                hist = 0
                if y_check not in multi_history[t]:
                    multi_history[t][y_check] = [0]
                max_pot = cumu_pot[t-1, hist] + pair_pot[hist, y_check]
                for prev_y in range(1, self.n_class):
                    prev_pot = cumu_pot[t-1, prev_y] + pair_pot[prev_y, y_check]
                    if prev_pot > max_pot:
                        max_pot = prev_pot
                        hist = prev_y
                        multi_history[t][y_check] = [prev_y]
                    elif abs (prev_pot - max_pot) <= tolerance:
                        multi_history[t][y_check].append(prev_y)
                cumu_pot[t, y_check] = max_pot + x_pots_costs[y_check]
                history[t, y_check] = hist
                
        def backtrack_multi_path(multi_history, t, y):
            if t == 1: return multi_history[t][y]
            ret = []
            for yp in multi_history[t][y]:
                prev = backtrack_multi_path(multi_history, t-1, yp)
                for path in prev:
                    ret.append(path + [y])
            return ret
        actions = []
        c = np.argmax(cumu_pot[-1])
        for y in range(self.n_class):
            if abs(c - cumu_pot[-1][y]) <= tolerance:
                actions.extend (backtrack_multi_path(multi_history, T-1, y) )
        # argmax
        # print ('multi history\n', multi_history)
        # print ('actions:\n')
        # for row in actions: print (row)
        # print ('edge potentials:\n', pair_pot)
        # print ('cumulative potentials:')
        # for row in cumu_pot.T:
        #     print (row)
        c = np.argmax(cumu_pot[-1])
        y_check = [-1] * T
        y_check[T-1] = c 
        for t in range(T-1, 0, -1):
            c = history[t, c] # backtrack target from this step
            y_check[t-1] = c
        # print (np.max(cumu_pot[-1]), y_check)
        return np.max(cumu_pot[-1]), y_check


    def _pcheck_singleoracle(self, sequence,  pcheck_actions, theta, transition_theta):
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
        # min_ -sum v_t - potentials(pcheck)
        # s.t. sum pchecks = 1 # A_eq = b_eq, one constraint
        # v_t <= C[yhat(t)] * pchecks
        #   v_t - C[yhat(t)] * pchecks <= 0, for all t, for all y_hat
        # -pcheck <= 0 for all pcheck in pchecks

        x = sequence
        n_action = len(pcheck_actions)
        T = len(sequence)

        objective = cvxopt.matrix([-1.] * (T) + [0.] * n_action) 
        
        # A_eq = b_eq
        A_eq = 1 + objective.T # all v's are 0, p_actions 1, so 1 + obj = 0... 1,1,1..
        b_eq = cvxopt.matrix(1.)
        
        A_ub = cvxopt.spmatrix([], [], [], (T * self.n_class + n_action, T + n_action) )

        for t in range(T):
            for i in range(n_action):
                objective[T + i] -= np.dot(x[t], theta[:, pcheck_actions[i][t]])
                if t > 0: # pairwise links
                    objective[T + i] -= transition_theta[pcheck_actions[i][t-1], pcheck_actions[i][t]]

            # A_ub <= b_ub
            # v_t - C[yhat(t), ychecks[t]] <= 0
            for yhat in range(self.n_class):
                # v_t for each phat
                row = t * self.n_class + yhat
                A_ub[row, t] = 1 # v_t
                # - C[yhat(t), ychecks[t]] for each ycheck actions, after T variables
                for i in range(n_action):
                    if self.cost_matrix[yhat, pcheck_actions[i][t]]:
                        A_ub[row, T + i] = -self.cost_matrix[yhat, pcheck_actions[i][t]]

        # -pcheck <= 0 for all pcheck in pchecks
        for i in range(n_action):
            A_ub[T * self.n_class + i, T + i] = -1

        # b_ub is 0 for all
        b_ub = cvxopt.matrix([0.] * (T * self.n_class + n_action) )

        # solve it
        # cvxopt.solvers.options['glpk'] = {'msg_lev' : 'GLP_MSG_OFF'}
        res = cvxopt.solvers.lp(objective, A_ub, b_ub, A_eq, b_eq, solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
        if res['status'] != 'optimal':
            print(res)
            exit
        gamevalue, pcheck_dist = -res['primal objective'], np.array(res['x'])[T:, 0] 
        return gamevalue, pcheck_dist


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
