'''
Created on May 8, 2018

@author: kaiser
'''
import numpy as np
from scipy.optimize import linprog

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

        if Cx.shape != (self.nrows, self.ncols):
            self._initialize(Cx.shape) # for single oracle, every call is different size
    
        # negative matrix cannot be solved always, but this modification doesn't change distribution
        C = Cx.copy()
        neg = np.amin(C) 
        if neg <= 0: C += (1 - neg)
        
        res = linprog(self.objective, A_ub=np.negative(C), b_ub=self.b, bounds=(self.lb, self.ub))
        
        while type(res.x) is not np.ndarray: 
            print(C)
            # C = C * 0.5
            res = linprog(self.objective, A_ub=np.negative(C), b_ub=self.b, bounds=(self.lb, self.ub), options=dict(bland=True, tol=1e-10/C.max()))
        
        v = 1 / sum(res.x)
        p_check = res.x * v
        
        if neg <= 0: v -= (1 - neg)

        return v, p_check
        
        
    def getRowMinimizerDist(self, C):
    
        v, p_hat = self.getColumnMaximizerDist(np.negative(np.transpose(C)))
    
        return -v, p_hat    
            
    def getRowMinimizerDistLP(self, Cx):

        C = Cx.copy()    
        neg = np.amin(C) 
        if neg <= 0: C += (1 - neg)
        
        res = linprog(self.b, A_ub=np.transpose(C), b_ub=self.objective, bounds=(self.lb, self.ub))
        
        if type(res.x) is not np.ndarray: 
            print(C)
            res = linprog(self.b, A_ub=np.transpose(C), b_ub=self.objective, bounds=(self.lb, self.ub), options=dict(bland=True, tol=1e-10/C.max()))
        
        v = 1 / sum(res.x)
        p_hat = res.x * v
        
        if neg <= 0: v -= (1 - neg)

        return v, p_hat
        