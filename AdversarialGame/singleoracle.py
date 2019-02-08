import numpy as np
from scipy.optimize import linprog

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

    def __init__(self, feature_theta, transition_theta, n_class, max_itr = 10000):
        """
        Initialize SingleOracle with the references to the 
        Theta parameters learned in the optimization.
        They are used in finding the best reponses.
        Parameters:
        -----------
            feature_theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
            n_class : number of target classes
        """
        self.feature_theta = feature_theta
        self.transition_theta = transition_theta
        self.n_class = n_class
        self.max_itr = max_itr


    def solve_p_hat_p_check(self, sequence):
        """
        The Single Oracle method

        Parameters:
        -----------
            sequence: the X or feature values of the sequence
        
        Returns:
        --------
            v : Number : game value
            p_hat : numpy array of size T x Y 
            p_check joints : numpy array of size T x Y x Y
            p_check marginals : numpy array of size T x Y
        """

        T = len(sequence)

        # initialize actions with all same class for full sequence for all classes
        pcheckactions = np.zeros((self.n_class, T))
        for i in range(n_class): pcheckactions[i, :] = i
        
        # # construct the lp and solve for P_hat
        # """ Minimize: c^T * x
        #         Subject to: A_ub * x <= b_ub
        #         A_eq * x == b_eq 
        # """
        # objective, A_ub, b_ub, A_eq, b_eq, lb, ub = self.initialize_singleoracle_phat_lp(T)
        # for action in pcheckactions: 
        #     self.add_singleoracle_constraint_for_pcheckaction(sequence, A_ub, b_ub, action)

        # use cvxopt for sparse matrices
        # http://cvxopt.org/userguide/coneprog.html#linear-programming 
        # but sparse/dense matrices in cvxopt cannot be changed in size
        # sparse matrix cannot shrink by making some entries 0

        # start the loop of solving and finding the best response
        previous_min_gamevalue = None
        loop_count, reached_stop = 0, False
        while (loop_count < self.max_itr and not reached_stop):
            loop_count += 1

            # solve the lp
            res = linprog(objective, A_ub, b_ub, A_eq, b_eq, bounds=(lb, ub))
            phat = res.x[1:] # variables are: v, phat11, phat12, phat21,..., phatTY
            min_gamevalue = res.x[0] # v is the game value / objective value, since all other objective coeff are 0
            
            if min_gamevalue == previous_min_gamevalue:
                reached_stop = True
                break
            previous_min_gamevalue = min_gamevalue

            # call best response function
            new_action_val, new_action = self.find_singleoracle_best_pcheck(sequence, phat)

            # if value is less than the p-hat solution, break
            if new_action_val <= min_gamevalue: break # adversary didn't find a choice to maximize cost

            # add best response to the actions, repeat
            if ( self.newpcheckaction_not_exist(new_action, pcheckactions) ):
                pcheckactions = np.append(pcheckactions, new_action, axis=0)
                self.add_singleoracle_constraint_for_pcheckaction(sequence, A_ub, b_ub, new_action)

        # after breaking the loop, call for pcheck solution
        max_gamevalue, pcheck_dist = self.pcheck_singleoracle(sequence,  pcheckactions)

        # the pcheck is for the ycheck actions, 
        # compute marginals and pairwise-joints 
        pairwise_pcheck, marginal_pcheck = self.compute_marginal_and_pairwise(pcheck_dist, T)

        return previous_min_gamevalue, phat, pairwise_pcheck, marginal_pcheck

    def solve_for_p_check(self, sequence):
        pass

if "__main__" == __name__:
    import numpy as np
    n_class = 3
    T = 10
    n_feature = 5
    theta = np.array([1, 2, -1, -2, 0])
    transition_theta = np.random.randint(-5, 5, (3, 3))
    x = np.random.randint(-5, 5, (T, n_feature))
    # print(theta, transition_theta, x)

    so = SingleOracle(theta, transition_theta, n_class)
    so.solve_p_hat_p_check(x)
