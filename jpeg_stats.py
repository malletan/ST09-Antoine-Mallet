import numpy as np
from matplotlib import pyplot as plt
from scipy.special import kv, modstruve, gamma
import NelderMead as nm
from scipy.stats import norm
import warnings




def getAC(data, x, y):
    """Return a list of the AC coefficients of same mod (x,y).
       Takes in input the array-like (single channel) dataset and extracts the AC coefficients.
       The block size is hard-coded and is equal to 8."""
    AC = np.asarray(data[x::8,y::8].flatten())
    return AC


class DCTstat:
    """The purpose of this class is to build the model of dct coefficients given in:
       'Statistical Model of Quantized DCT Coefficients: Application in the Steganalysis of Jsteg Algorithm'."""

    def __init__(self, data, Q, x, y, payload=None):
        if isinstance(data, np.ndarray):
            """A few input types are handled in the following. Data can either be the vector as a 1-D ndarray or a list, or can be the full dct matrix.
            In the latter case, x and y paramaters are used to extract the mode (x,y). The field 'data' is of list type."""
            if len(data.shape) == 1:
                self.data = data
            elif len(data.shape) == 2:
                self.data = getAC(data, x, y)
        elif isinstance(data, list):
            self.data = np.asarray(data)
        
        self.x = x
        self.y = y

        if isinstance(Q, np.ndarray):
            """The quantization step can either be given directly (then of type int), 
               or the whole matrix can be given, we then extract the (x,y) step"""
            self.Q = Q[x][y]
        else:
            self.Q = Q
        
        self.payload = payload # Embedding Rate
        self.nk = len(self.data) - np.sum(np.array(self.data) == 0) - np.sum(np.array(self.data) == 1 )

        self.eta = self._eta(self.data)
        self.nu = self._nu(self.data)

        self.lower_bound = int(np.floor(  np.min(self.data) /2 ) * 2)
        self.upper_bound = int(np.floor( np.max(self.data) /2 ) * 2 +1)
        self.bins = self.upper_bound - self.lower_bound
        
        self.X = np.linspace(self.lower_bound, self.upper_bound, self.bins*100) # an interval of real values. Used to compute the pdf
        self.L = np.arange(self.lower_bound, self.upper_bound+1).astype('int') # an interval of integer values. Used to compute the pmf

        self.param_optimized = False
        self.qstep_optimized = False

    
    def set_payload(self, payload):
        self.payload = payload


    def compute(self):
        """The method creates and computes the pdf and pmf of the statistical dataset."""
        self.Fi = self.FI(self.X, self.eta, self.nu)
        self.pmf = self.Pv(self.L)


    def moment(self, data, n):
        """Returns the n-th moment of the dataset."""
        m = 0
        for d in data:
            m += int(d)**n # d is an element of an ndarray, which type is np.int32 by default. 
                           # It therefore can not handle numbers larger than 2**32 - 1, which can be met for the lowest frequencies. 
        return (m / len(data))

 
    def _eta(self, data):
        """Returns the eta parameter of the dataset distribution."""
        m2 = self.moment(data, 2)
        m4 = self.moment(data, 4)
        s0 = 1/12
        if (m2 < 0.5 ):
            s0 = np.max( [ 0, norm.cdf((m2-0.212)*9)*1/12  -0.004 + 0.0305 * m2 -0.0394* m2**2 - 0.01 * m2**3 ] )
        tmp = np.max( [ 0.1 , ((m2 - s0)**2) / (1/3 * m4 - m2**2 + 1/360) ] )

        return tmp


    def _nu(self, data):
        """Returns the nu parameter of the dataset distribution."""
        m2 = self.moment(data, 2)
        s0 = 1/12
        if (m2 < 0.5 ):
            #print("Second moment inferior to .5")
            s0 = np.max( [ 0, norm.cdf((m2-0.212)*9)*1/12  -0.004 + 0.0305 * m2 -0.0394* m2**2 - 0.01 * m2**3 ] )
        tmp = np.max( [ 0.1 ,  (self.Q**2) * (m2 - s0) / self._eta(data) ] )

        return tmp
    

    def FI(self, X, eta, nu):
        """Computes the pdf of the distribution. X is expected to be either a linear space or a single point."""
        return np.sqrt(2/np.pi) * (np.abs(X) * np.sqrt(nu/2))**(eta-.5) * kv(eta-.5, np.abs(X) * np.sqrt(2/nu)) / (nu**eta * gamma(eta))


    def g(self, l, eta=None, nu=None):
        if l is None:
            l = self.L
        if eta is None:
            eta = self.eta
        if nu is None:
            #print("Nu=None en paramètre", nu)
            nu = self.nu
            #print("Après gestion:", nu)
        #print("l pour calculer g:", l)
        #print("nu pour calculer g:", nu)
        warnings.filterwarnings('error')
        try:
            tmp = np.sqrt(2/nu) * self.Q * (l+.5)
        except RuntimeWarning:
            #We still want to do the computation,
            warnings.filterwarnings('ignore')
            tmp = np.sqrt(2/nu) * self.Q * (l+.5)
        warnings.filterwarnings('default')
            #print("g = ", tmp)
            #print("Subband({},{})".format(self.x, self.y))
            #But we can now identify the origin of the problem..
            #print("Nu parameter: ", nu)

            
        return tmp 


    def G(self, l=None, e=None, n=None):
        """Computes the positive part of the Probability Mass Function Pv(l) expressed in Eq (28).
           The sub-function g(l) is first computed following Eq (31).
           Since Pv is symetrical, we compute the positive half, and then reverse it."""
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        #print("ça calcule gl")
        gl = self.g(l)
        #print("ça a calculé gl")
        warnings.filterwarnings('error')
        try:
            #print("\tça calcule dans le try")
            tmp = (kv(e-.5, gl) * modstruve(e-1.5, gl) + kv(e-1.5, gl) * modstruve(e-.5, gl)) * .5 * gl
            #print("\tça a calculé dans le try")
        except RuntimeWarning:
            warnings.filterwarnings('ignore')
            tmp = (kv(e-.5, gl) * modstruve(e-1.5, gl) + kv(e-1.5, gl) * modstruve(e-.5, gl)) * .5 * gl
        warnings.filterwarnings('default')
        #print( len( tmp[ np.isnan(tmp) == True ] ) )

        return tmp


    def _pv(self, l, e=None, n=None):
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        pmf = np.zeros(l.shape)
        warnings.filterwarnings('error')
        try:
            pmf = self.G(abs(l), e, n) - self.G(abs(l) - 1, e, n)
        except RuntimeWarning as w:
            print(w)
            warnings.filterwarnings('ignore')
            pmf = self.G(abs(l), e, n) - self.G(abs(l) - 1, e, n)
            #print("pmf: ", pmf)

        warnings.filterwarnings('default')
        pmf[l==0] = 2*self.G(np.int32(0), e, n)
        return pmf


    def Pv(self, l=None, e=None, n=None, Q=None):
        """Computes the Probability Mass Function given in Eq (32).
        Parameter:
        l: The values in the histogram."""
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        if Q is not None:
            #print("Before quantization:", len(self.data))
            """
            warnings.filterwarnings('error')
            try:
                print("On try")
                div = self.data / Q
                print("div")
                diq = (div).astype('int')
                print("diq")
                mask = ((div - diq).astype('bool') * -1 + 1).astype('bool')
                print("mask")
                self.data = self.data[mask]
                print("new_data")
            except RuntimeWarning as w:
                print(w)
                warnings.filterwarnings('ignore')
                div = self.data / Q
                diq = (div).astype('int')
                mask = ((div - diq).astype('bool') * -1 + 1).astype('bool')
                self.data = self.data[mask]
                print( len( self.data[ np.isnan(self.data) == True ] ) )
            print("After quantization:", len(self.data))

            warnings.filterwarnings('default')
            """
            #print("Q:", Q)
            #print("self.Q:", self.Q)
            div = self.data / Q
            #print(div)
            diq = (div).astype('int')
            #print(diq)
            mask = ((div - diq).astype('bool') * -1 + 1).astype('bool')
            #print(mask)
            self.data = self.data[mask]
            self.lower_bound = int(np.floor(  np.min(self.data) /2 ) * 2)
            self.upper_bound = int(np.floor( np.max(self.data) /2 ) * 2 +1)
            self.L = np.arange(self.lower_bound, self.upper_bound+1).astype('int') # an interval of integer values. Used to compute the pmf
            #print(self.data)
    
        return self._pv(l, e, n)


    def _qr(self, l, payload=None, e=None, n=None):
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        pmfStego = np.zeros(l.shape)
        warnings.filterwarnings('error')
        try:
            pmfStego = (1 - payload/2) * self._pv(l, e, n) + payload / 2 * self._pv(l+(-1)**np.abs(l), e, n)
        except RuntimeWarning:
            warnings.filterwarnings('ignore')
            pmfStego = (1 - payload/2) * self._pv(l, e, n) + payload / 2 * self._pv(l+(-1)**np.abs(l), e, n)
            #print("pmf Stego: ", pmfStego)

        warnings.filterwarnings("default")
        pmfStego[(l==0) | (l==1)] = self._pv( np.array([0,1]), e, n)
        return pmfStego


    def Qr(self, payload=None, l=None, e=None, n=None):
        """Computes the Probability Mass Function under hypothesis H1.
           A regular l is expected, as the function starts by the LSB flipping operation"""
        if payload is None:
            payload = self.payload
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        return self._qr(l, payload, e, n)


    def _lr(self):
        Pv = self.Pv()
        Qr = self.Qr()
        warnings.filterwarnings('error')
        try:
            tmp = np.divide( Qr , Pv , out=np.zeros_like(Qr), where=Pv!=0) 
        except RuntimeWarning:
            warnings.filterwarnings('ignore')
            tmp = np.divide( Qr , Pv , out=np.zeros_like(Qr), where=Pv!=0)
            print("log ratio: ", tmp)

        warnings.filterwarnings('default')
        return tmp


    def _llr(self):
        return np.ma.filled(np.ma.log( self._lr() ), 0 )


    def logRatio(self):
        LR = self._lr()
        dct = self.data
        idxNOT01 = (dct!=0) & (dct!=1)
        dctNOT01 = dct[idxNOT01]
        data = (dctNOT01 - self.lower_bound).astype('int')
        return np.nansum( np.ma.filled(np.ma.log( LR[data] ), 0 ) )


    def logLikelyhood(self, e=None, n=None, Q=None):
        """Used to optimize the parameters"""
        if e is not None and n is not None:
            self.eta = e
            self.nu  = n
        PMF = self.Pv(e=e, n=n, Q=Q)
        data = (self.data - self.lower_bound).astype('int')
        return -1*np.nansum( np.ma.filled(np.ma.log( PMF[data] ), 0 ) )


    def optimize_parameters(self):
        optim = nm.ParamOptimizer(self)
        self.eta = optim.opt.x[0]
        self.nu = optim.opt.x[1]
        
        self.param_optimized = True


    def optimize_quant_step(self, candidates):
        if not self.param_optimized:
            self.optimize_parameters()
        #print("Nu after optimization:", self.nu)
        #print("Optimisation lancée")
        optim = nm.QStepOptimizer(self, candidates)
        #print("Optimization réussie")
        return optim.Q


    def espk(self):
        LLR = self._llr()
        warnings.filterwarnings('error')
        try:
            mu = np.nansum( LLR * self.Pv() )
        except RuntimeWarning:
            warnings.filterwarnings('ignore')
            mu = np.nansum( LLR * self.Pv() )
            print("mu: ", mu)
        warnings.filterwarnings('default')
        self.mu = mu


    def vark(self):
        LLR = self._llr()
        warnings.filterwarnings('error')
        try:
            sig = np.nansum( (LLR - self.mu)**2 * self.Pv() )
        except RuntimeWarning:
            warnings.filterwarnings('ignore')
            sig = np.nansum( (LLR - self.mu)**2 * self.Pv() )
            print(sig)
        warnings.filterwarnings('default')
        self.sig = sig

    def esp0(self):
        self.espk()
        self.vark()
        n = len(self.data)
        pk = 1 - np.sum( self._pv( np.array([0,1]) ) )
        if np.isnan( pk ):
            pk = 0
#        print(' n = ' , n , ' pk = ' , pk , ' mu = ' , self.mu  , ' var = ' , self.sig )
        return n * pk * self.mu
    

    def var0(self):
        n = len(self.data)
        pk = 1 - np.sum( self._pv( np.array([0,1]) ) )
        if np.isnan( pk ):
            pk = 0
        return n * pk * self.sig + n * pk * (1 - pk) * self.mu**2
    




"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import kv, modstruve, gamma
import scipy.optimize as optimize
import NelderMead as nm
import warnings


def getAC(data, x, y):
    Return a list of the AC coefficients of same mod (x,y).
       Takes in input the array-like (single channel) dataset and extracts the AC coefficients.
       The block size is hard-coded and is equal to 8.
    AC = []
    for i in range(0, data.shape[0], 8):
        for j in range(0, data.shape[1], 8):
            AC.append(data[i+x][j+y])
    return AC


class DCTstat:
    The purpose of this class is to build the model of dct coefficients given in:
       'Statistical Model of Quantized DCT Coefficients: Application in the Steganalysis of Jsteg Algorithm'.

    def __init__(self, data, Q=None, x=None, y=None, payload=None):
        if isinstance(data, np.ndarray):
            A few input types are handled in the following. Data can either be the vector as a 1-D ndarray or a list, or can be the full dct matrix.
            In the latter case, x and y paramaters are used to extract the mode (x,y). The field 'data' is of list type.
            if len(data.shape) == 1:
                self.data = data.tolist() 
            elif len(data.shape) == 2:
                self.data = getAC(data, x, y)
        elif isinstance(data, list):
            self.data = data
        
        #The following describes the subband coordinates in the 8x8 blocks. Without any usage in this version of the code, 
        #expect for extracting the subband and the associated step.
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

        if Q is None:
            Q can be omitted when dealing with unquantized coefficients as in paper 1.
            self.eta = self.eta_unquantized(self.data)
            self.nu = self.nu_unquantized(self.data)

        else:
            if isinstance(Q, np.ndarray):
                The quantization step can either be given directly (then of type int), 
                or the whole matrix can be given, we then extract the (x,y) step
                self.Q = Q[x][y]
            else:
                self.Q = Q
        
            self.payload = payload # Embedding Rate
            self.nk = len(self.data) - np.sum(np.array(self.data) == 0) - np.sum(np.array(self.data) == 1 )

            self.eta = self._eta(self.data)
            self.nu = self._nu(self.data)



        self.lower_bound = min(self.data)
        self.upper_bound = max(self.data) 
        self.bins = self.upper_bound - self.lower_bound
        
        self.X = np.linspace(self.lower_bound, self.upper_bound, self.bins*100) # an interval of real values. Used to compute the pdf
        self.L = np.asarray(range(self.lower_bound, self.upper_bound+1)) # an interval of integer values. Used to compute the pmf

    
    def set_payload(self, payload):
        self.payload = payload


    def compute(self):
        The method creates and computes the pdf and pmf of the statistical dataset.
        self.Fi = self.FI(self.X, self.eta, self.nu)
        self.pmf = self.Pv(self.L)


    def moment(self, data, n):
        Returns the n-th moment of the dataset.
        m = 0
        for d in data:
            m += int(d)**n # d is an element of an ndarray, which type is np.int32 by default. 
                           # It therefore can not handle numbers larger than 2**32 - 1, which can be met for the lowest frequencies. 
        return (m / len(data))

 
    def _eta(self, data):
        Returns the eta parameter of the dataset distribution.
        m2 = self.moment(data, 2)
        m4 = self.moment(data, 4)
        return ((m2 - 1/12)**2) / (1/3 * m4 - m2**2 + 1/360)


    def _nu(self, data):
        Returns the nu parameter of the dataset distribution.
        m2 = self.moment(data, 2)
        return (self.Q**2) * (m2 - 1/12) / self._eta(data)


    def eta_unquantized(self, data):
        m2 = self.moment(data, 2)
        m4 = self.moment(data, 4)
        return 3 / ( (m4 / m2**2) - 3)


    def nu_unquantized(self, data):
        m2 = self.moment(data, 2)
        return m2 / self.eta


    def FI(self, X=None, e=None, n=None):
        Computes the pdf of the distribution. X is expected to be either a linear space or a single point.
        if X is None:
            X = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu

        return np.sqrt(2/np.pi) * (np.abs(X) * np.sqrt(n/2))**(e-.5) * kv(e-.5, np.abs(X) * np.sqrt(2/n)) / (n**e * gamma(e))

    def g(self, l):
        return np.sqrt(2/self.nu) * self.Q * (l+.5)


    def G(self, l=None, e=None, n=None):
        Computes the positive part of the Probability Mass Function Pv(l) expressed in Eq (28).
           The sub-function g(l) is first computed following Eq (31).
           Since Pv is symetrical, we compute the positive half, and then reverse it.
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        if ( (isinstance(l, np.int32) or isinstance(l, int)) and l >= 0 ) or ( isinstance(l, np.ndarray) and all(l >= 0) ):
            gl = self.g(l) 
            return (kv(e-.5, gl) * modstruve(e-1.5, gl) + kv(e-1.5, gl) * modstruve(e-.5, gl)) * .5 * gl
            #return (kv(self.eta-.5, g(l)) * modstruve(self.eta-1.5, g(l)) + kv(self.eta-1.5, g(l)) * modstruve(self.eta-.5, g(l))) * .5 * g(l)
        else:
            print("Error: input should be positive instead of {0} of type {1}".format(l, type(l)))
            return 0


    def _pv(self, l, e=None, n=None):
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        if l == 0:
            return 2*self.G(np.int32(0), e, n)
        else:
            return self.G(abs(l), e, n) - self.G(abs(l) - 1, e, n)


    def Pv(self, l=None, e=None, n=None):
        Computes the Probability Mass Function given in Eq (32).
        Parameter:
        l: The values in the histogram.
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        pv = []
        for i in l:
            pv.append( self._pv(i, e, n) )
        return pv


    def _qr(self, l, payload=None, e=None, n=None):
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu

        if l == 0:
            return self._pv(l, e, n)
        if l == 1:
            return self._pv(l, e, n)
        else:
            return (1 - payload/2) * self._pv(l, e, n) + payload / 2 * self._pv(l+(-1)**np.abs(l), e, n)


    def Qr(self, payload=None, l=None, e=None, n=None):
            Computes the Probability Mass Function under hypothesis H1.
            A regular l is expected, as the function starts by the LSB flipping operation
        if payload is None:
            payload = self.payload
        if l is None:
            l = self.L
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu

        qr = []
        for i in l:
            qr.append( self._qr(i, payload, e, n) )
        return qr


    def _lr(self, m):
        return np.log(self._qr(m, self.payload) / self._pv(m))


    def logRatio(self):
        lr = 0
        for d in self.data[:self.nk]:
            lr += self._lr(d)
        #self.log_ratio = lr
        return lr


    def logLikelyhood(self, param):
        
            Computes the inverse of a probability function. Either computes the PDF or the PMF.
            We want to compute the loglikelihood to then optimize it to get the parameters' ML estimates.
            In the case of quantized coefficients (e.g. paper 0), we should use the PMF, thus the built-in Pv function.
            In the case of unquantized coefficients (e.g. paper 1), we should use the PDF, thus the built-in FI function. 
        
        VAL = self.optimize_func(e=param[0], n=param[1]) #optimize_function needs to be set beforehand.
        data = np.array(self.data - self.lower_bound)
        LL = 0
        for i in data:
            tmp = np.log( VAL[i] )
            if np.abs(tmp) != np.inf:
                LL = LL - tmp
        return LL


    def optimize_parameters(self, fct):
        x0 = [ self.eta, self.nu ]
        self.optimize_func = fct
        optim = optimize.minimize(self.logLikelyhood, x0, method="Nelder-Mead")
        self.etaML = optim.x[0]
        self.nuML = optim.x[1]
        return optim


    def espk(self):
        mu = 0 
        for m in self.L:
            mu = mu + self._lr(m) * self._pv(m)
        self.mu = mu
        
    
    def vark(self):
        sig = 0
        for m in self.L:
            sig = sig + (self._lr(m) - self.mu)**2 * self._pv(m)
        self.sig = sig
    
    def esp0(self):
        self.espk()
        self.vark()
        n = len(self.data)
        pk = 1 - self._pv(0) - self._pv(1)
        return n * pk * self.mu
    

    def var0(self):
        n = len(self.data)
        pk = 1 - self._pv(0) - self._pv(1)
        return n * pk * self.sig + n * pk * (1 - pk) * self.mu**2
    





"""









"""

    def PvRC(self, e=None, n=None):
        if e is None:
            e = self.eta
        if n is None:
            n = self.nu
        allValues = self.L
        Q = self.Q
        pv=[]
        for l in allValues:
            if l == 0:
                g0 = np.sqrt(2/n) * Q * (np.abs(0)+.5)
                val = (kv(e-.5, g0) * modstruve(e-1.5, g0) + kv(e-1.5, g0) * modstruve(e-.5, g0)) * .5 * g0
                pv.append(2*val)
            else:
                gl = np.sqrt(2/n) * Q * (np.abs(l)+.5)
                glm1 = np.sqrt(2/n) * Q * (np.abs(l)-.5)
                val= (kv(e-.5, gl) * modstruve(e-1.5, gl) + kv(e-1.5, gl) * modstruve(e-.5, gl)) * .5 * gl
                valm1= (kv(e-.5, glm1) * modstruve(e-1.5, glm1) + kv(e-1.5, glm1) * modstruve(e-.5, glm1)) * .5 * glm1
                pv.append(val - valm1)
        return pv

"""