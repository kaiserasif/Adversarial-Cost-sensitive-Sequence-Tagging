import numpy as np
import cvxopt

class ZerosumGame:

    def __init__(self, nrows, ncols):
        self._initialize(nrows, ncols)  

    def _initialize(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.objective = np.ones(self.ncols)
        self.b = np.negative(np.ones(self.nrows))
        self.lb = 0
        self.ub = None  
        
    def getColumnMaximizerDist(self, Cx):
        """
        Solve for column maximizer distribution
        If passed numpy array, converts it first to cvxopt.matrix
        Parameter:
            Cx : Augmented game : Preferable cvxopt.matrix object, otherwise numpy array
        Returns:
            v : Game value : double
            p_check : column distribution : type of Cx - matrix or ndarray
        """

        isndarray = type(Cx) is np.ndarray
        if isndarray:
            C = cvxopt.matrix(Cx)
        else:
            C = Cx.copy()

        # negative matrix cannot be solved always, but this modification doesn't change distribution
        # neg = np.amin(C) 
        # if neg <= 0: C += (1 - neg)
        
        # res = linprog(self.objective, A_ub=np.negative(C), b_ub=self.b, bounds=(self.lb, self.ub))
        
        # while type(res.x) is not np.ndarray: 
        #     print(C)
        #     # C = C * 0.5
        #     res = linprog(self.objective, A_ub=np.negative(C), b_ub=self.b, bounds=(self.lb, self.ub), options=dict(bland=True, tol=1e-10/C.max()))
        
        # v = 1 / sum(res.x)
        # p_check = res.x * v
        
        # if neg <= 0: v -= (1 - neg)

        return v, p_check if not isndarray else np.array(p_check)[:, 0]
        
        
    def getRowMinimizerDist(self, C):
    
        v, p_hat = self.getColumnMaximizerDist(np.negative(np.transpose(C)))
    
        return -v, p_hat    
            
    def getRowMinimizerDistLP(self, Cx):
        pass
        # C = Cx.copy()    
        # neg = np.amin(C) 
        # if neg <= 0: C += (1 - neg)
        
        # res = linprog(self.b, A_ub=np.transpose(C), b_ub=self.objective, bounds=(self.lb, self.ub))
        
        # if type(res.x) is not np.ndarray: 
        #     print(C)
        #     res = linprog(self.b, A_ub=np.transpose(C), b_ub=self.objective, bounds=(self.lb, self.ub), options=dict(bland=True, tol=1e-10/C.max()))
        
        # v = 1 / sum(res.x)
        # p_hat = res.x * v
        
        # if neg <= 0: v -= (1 - neg)

        # return v, p_hat
        