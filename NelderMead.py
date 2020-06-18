import numpy as np
import jpeg_stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize
import os
from scipy.optimize import Bounds
import time


class ParamOptimizer:
    def __init__(self, statObject):
        #print("---OPTIMIZATION BEGINS---")
        #print(statObject.x, statObject.y)
        self.stat = statObject
        self.x0 = np.asarray([self.stat.eta, self.stat.nu])
 #       b = Bounds([0.001 , 1 ] , [100 ,  100000])
 #       self.opt = optimize.minimize(self.objective, self.x0, method='trust-constr', options={'maxiter' : 15}, bounds=b )
        
        self.opt = optimize.minimize(self.objective, self.x0, method='Nelder-Mead', options={'maxiter' : 10} )

        if (self.opt.x[0] <= 0 or self.opt.x[1] <= 0 or np.isinf( self.opt.x[0] ) or np.isinf( self.opt.x[1] ) or np.isnan( self.opt.x[0] ) or np.isnan( self.opt.x[1] ) ):
            print(' /!\\ Search for optimal distribution parameters failed /!\\ Switching to "Trust Region" optimization method /!\\ ')
            b = Bounds([0.001 , 0.1 ] , [10000 ,  1000000])
            self.opt = optimize.minimize(self.objective, self.x0, method='trust-constr', options={'maxiter' : 20}, bounds=b )


    def objective(self, x):
        """Equation (41). In the problem Pv is written as Pv a function of V_ki and 
           then optimized following eta_k and nu_k, which are then considered the variables.
           In the optimization objective function, Pv is thus regarded as a function of the parameters.
           But in the jpeg_stats' DCTstat class, Pv is still computed as shown in Eq (32)."""
        return (self.stat.logLikelyhood(e=x[0], n=x[1]))



class QStepOptimizer:

    def __init__(self, statObject, Qcandidates):
        #print("---QStep OPTIMIZATION BEGINS---")
        self.stat = statObject
        #print("Nu parameter in the optimizer:", self.stat.nu)
        #print("The candidates are:", Qcandidates)
        ll = self.objective(Qcandidates)
        #print("Print the loglikelyhoods are:\n", ll)
        self.Q = Qcandidates[np.argmax(ll)]

    def objective(self, candidates):
        qss = []
        for q in candidates:
            qss.append(self.stat.logLikelyhood(Q=q))
        
        return qss
