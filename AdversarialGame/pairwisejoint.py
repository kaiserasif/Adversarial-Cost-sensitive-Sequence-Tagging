from .pairwisejointlp_cvxopt import PairwiseJointLPSovler as Cvxsolver
from .pairwisejointlp_cvxopt import PairwiseJointLPSovler as Gurobisolver

class PairwiseJoint:
    """
    Solve sequence tagging problem using
    pariwise-marginal formulation
    in contrast to Single Oracle method.

    Predictor's distribution isn't needed during training,
    therefore, those are not learned. Uses a LP of 
    T + (T-1) * n_class variables
    T * v_t inequality constratns for each of T phats
    and additional T-2 constraints for pairwise pcheck's 
    equlity constraints. 
    Number of variables are initially larger than Single Oracle
    But one lp to solve, Single Oracle needs to incrementally
    increase variables.  
    """

    def __init__(self, n_class, cost_matrix, solver_type):
        """
        Initialize SingleOracle with the references to the 
        Theta parameters learned in the optimization.
        They are used in finding the best reponses.
        Parameters:
        -----------
            n_class : number of target classes
            cost_matrix : numpy matrix of n_class x n_class
            solver_type : string : 'gurobi' or 'cvxopt'
        """
        self.n_class = n_class
        self.cost_matrix = cost_matrix
        self.solver_type = solver_type

        self.solver = None


    def set_param(self, **kwargs):
        """
        this method provides the ability to set the updated 
        parameters
        """
        self.__dict__.update(kwargs)


    def solve_for_p_check(self, sequence, theta, transition_theta):
        """
        The Pairwise-Marginal method of solving p_check

        Parameters:
        -----------
            sequence: the X or feature values of the sequence
            theta : numpy matrix of size: n_feature x n_class 
            transition_theta : numpy matrix of size: n_class x n_class
        
        Returns:
        --------
            v : float : game value
            p_check joints : numpy array of size T x Y x Y
            p_check marginals : numpy array of size T x Y
        """
        T = len(sequence)
        
        if self.solver_type == 'gurobi' and not isinstance(self.solver, Gurobisolver):
                self.solver = Gurobisolver(self.n_class, self.cost_matrix)
        elif self.solver_type == 'cvxopt' and not isinstance(self.solver, Cvxsolver):
                self.solver = Cvxsolver(self.n_class, self.cost_matrix)
    
        # call solvers to get distributions
        gamevalue, vars = self.solver.solve_lp(sequence, theta, transition_theta)
        pairwise_pcheck = vars[T:]

        pairwise_pcheck, marginal_pcheck = self._compute_marginal_and_pairwise(pairwise_pcheck, T)
        
        return gamevalue, pairwise_pcheck, marginal_pcheck


    def _compute_marginal_and_pairwise(self, pairwise_pcheck, T):
        """
        from pcheck actions' probabilities
        compute the pairwise and marginal pchecks for each T
        Parameters:
            pairwise_pcheck : 1d ndarray T * n_class^2 length
            T : length of the sequence
        Returns:
            pairwise_pcheck : List[List[List[float]]] of y1y2, y2y3... size: T-1 x n_class x n_class
            marginal_pcheck : List[List[float]] of y1, y2, y3... size T x n_class
        """
        pcheks = pairwise_pcheck # save the reference

        pairwise_pcheck = [ [ [0]*self.n_class for _ in range(self.n_class) ] for _ in range(T-1) ] # T-1 pairs
        marginal_pcheck = [ [0]*self.n_class for _ in range(T) ]

        for i in range(0, T-1): # y1y2, y2y3 : starts from 2nd index
            for a in range(self.n_class):
                for b in range(self.n_class):
                    # var order pipj: (a=0, b=0), (a=0, b=1), (a=1, b=0), (a=1, b=1)
                    p = pcheks[i*(self.n_class**2) + a*self.n_class + b]
                    pairwise_pcheck[i][a][b] = p
                    marginal_pcheck[i][a] += p
        # final node
        for a in range(self.n_class):
            for b in range(self.n_class):
                # var order pipj: (a=0, b=0), (a=0, b=1), (a=1, b=0), (a=1, b=1)
                p = pcheks[(T-2)*(self.n_class**2) + a*self.n_class + b]
                marginal_pcheck[T-1][b] += p
                
        return pairwise_pcheck, marginal_pcheck
