"""
The following is 




"""
import jpegio as jio
import numpy as np
import dct as DCT
import jpeg_module as jmd
import jpeg_stats as jst
from scipy.stats import norm
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time


xlim = range(8)
ylim = xlim


range_q1 = range(1, 21) # We will be looking for the optimal q1 in this range


def kl_divergence(p, q):
    """Compute the Kullback-Leibler divergence between 2 pdf.
    
    -----------
    Parameters:
    
    p, q -- pdf of discrete random variables. Expexted to be 1-D numpy.ndarray.
    -----------
    """

    return np.sum(np.where(p != 0, p*np.log(p / q), 0))


def kl_distance(p, q):
    """Compute the Kullback-Leibler distance between 2 pdf.
    
    -----------
    Parameters:
    
    p, q -- pdf of discrete random variables. Expexted to be 1-D numpy.ndarray.
    -----------
    """

    eps = .00000001
    p += eps
    q += eps
    return kl_divergence(p, q) + kl_divergence(q, p)


def dct2vec(DCT):
    """Group together the coefficients of the same 8x8 base-image of an 8x8 Discrete Cosine Transform.
    
    -----------
    Parameters:
    
    DCT -- 2-D numpy.ndarray. 
    -----------
    """

    nbdct = DCT.shape
    dctVec = np.zeros( [ 64 , int(nbdct[0]*nbdct[1]/64) ] )
    for x in xlim:
        for y in ylim:
            if x != 0 or y != 0:
                dctVec[x*8+y,:] = DCT[x::8,y::8].flatten()
    
    return dctVec


def  get_hist(data, norm=False):
    """Create a dictionary which keys are the histogram's bins and values are the bins' heights.
    
    -----------
    Parameters:
    
    data -- Iterable-like variable whose values are to be used for constructing the histogram.

    norm (Optional) -- If set to True, the sum of the heights is equal to 1.
    -----------
    """

    tmp = {}
    minval = int(min(data))
    maxval = int(max(data))+1
    itt = 0 
    ott = 0 
    for b in range(minval, maxval):
        tmp[b] = 0
    for d in data:
        ott += 1 
        if d > minval and d < maxval:
            itt += 1
            tmp[d] += 1
    
    if norm is True:
        for v in tmp:
            tmp[v] /= ott
    
    return tmp


def cdfe(x1, x2):
    """Return the cumulative distribution function of the gaussian distribution of mean 0 and variance 1/12.

    -----------
    Parameters:

    x1, x2 -- 2 numbers between which the probability is computed. 
    -----------
    """

    return norm.cdf(x2, 0, np.sqrt(1/12)) - norm.cdf(x1, 0, np.sqrt(1/12))


even_subrange_proba = np.array([
    cdfe(-1.5, -1),
    cdfe(-1, 0),
    cdfe(0,1),
    cdfe(1,1.5)
])
#print("roundoff error probabilities for even q2:\n", even_subrange)

odd_subrange_proba = np.array([
    cdfe(-1.5, -.5),
    cdfe(-.5, .5),
    cdfe(.5, 1.5)
])
#print("roundoff error probabilities for odd q2:\n", odd_subrange)


def get_range(data, q2):
    """Return the range for all the secondary quantized coef as shown in Equation (19).
    
    -----------
    Parameters:

    data -- either 1-D array of integer or single integer. Corresponds to c2 in the equation.

    q2 -- Single integer. Secondary quantization step.  
    -----------
    """

    return (data * q2 + np.array( [ [- np.floor(q2/2) - 1], [np.floor(q2/2) +1] ] )).astype(np.int64)


def single_val_subrange_even(c2, q2):
    """Return the subranges of c2 and q2, given that c2 is an integer and q2 is even."""

    roundoff = np.arange(2,-2, -1)
    roundoff = np.array([roundoff, roundoff -1])
    subrange = np.ones((2,4)) * q2 * c2 + np.array([[-1,1]] * 4).swapaxes(0,1) * q2/2 + roundoff
    return subrange


def single_val_subrange_odd(c2, q2):
    """Return the subranges of c2 and q2, given that c2 is an integer and q2 is odd."""

    roundoff = np.arange(1,-2, -1)
    roundoff = np.array([roundoff, roundoff])
    subrange = np.ones((2,3)) * q2 * c2 + np.array([[-1,1]] * 3).swapaxes(0,1) * q2/2 + roundoff
    return subrange


def multiple_val_subrange(data, q2, range_builder=None):
    """Return the subranges for each value in data.
    
    -----------
    Parameters:

    data -- 1-D numpy.ndarray of DCT coefs.

    q2 -- Secondary quantization step.

    range_builder (Optional) -- should be either `single_val_subrange_odd` or `single_val_subrange_even` 
    -----------
    """

    if range_builder is None:
        if q2 % 2 == 0:
            range_builder = single_val_subrange_even
        else:
            range_builder = single_val_subrange_odd

    subranges = []
    for c2 in data:
        subranges.append(range_builder(c2, q2))

    return np.asarray(subranges)


def get_subrange(data, q2):
    """Return the subrange intervals for data, given q2. 

    -----------
    Parameters:

    data -- Contains DCT coefficients. Can be either an integer (int, np.int16, np.int32 or np.np.int64).

    q2 -- The secondary quantization step.

    -----------
    """

    if q2 % 2 == 0:
        range_builder = single_val_subrange_even

    else:
        range_builder = single_val_subrange_odd

    if type(data) == np.int64 or type(data) == np.int32 or type(data) == 'int' or type(data) == np.uint16 :
        return range_builder(data, q2)

    elif type(data) == np.ndarray:
        return multiple_val_subrange(data, q2, range_builder)
    
    elif type(data) == list:
        return multiple_val_subrange(np.asarray(data), q2, range_builder)


def single_proba(k, subrange):
    """Compute the probability p(k) as described in equation (20). 
    We compute the probability that k lies in the range Rq2(c2). Here Rq2(c2) is considered given. 

    The computation is the following:
    We substract the value k to the subrange array, and then multiply the new min and max. 
    If k is in the subrange, then the product will be negative, and it will be positive otherwise:
    
    if min < k and max < k:   min - k < 0 and max - k < 0 => (min - k) * (max - k) > 0

    if min > k and max > k:   min - k > 0 and max - K > 0 => (min - k) * (max - k) > 0
    
    if min < k and max > k:   min - k < 0 and max - K > 0 => (min - k) * (max - k) < 0

    From there, we can easily isolate the subranges from which we add the probabilities.
    
    -----------
    Parameters:

    k -- Is supposed to be an integer.
    
    subrange -- a 2-D array of shape (2,4) or (2,3), from which the corresponding probabilities (even or odd) will be summed.
    
    -----------
    """
    
    subrange_proba = [even_subrange_proba, odd_subrange_proba]
    in_range = subrange - k
    in_range = in_range[0] * in_range[1]
    pk = np.sum(subrange_proba[subrange.shape[1] % 2][in_range < 0])
    return pk


def subrange_proba(k, subrange):
    """Return pk as in equation (20). 
    This function deals with int and arrays as input for k. 
    It then relies on the `single_proba` function to do the proper computation1
    
    -----------
    Parameters:

    k -- either a single integer or an array of int to test on a  given subrange.

    subrange -- a 2-D array of shape (2,4) or (2,3)

    -----------
    """

    if type(k) == np.int16 or type(k) == np.int32 or type(k) == int: # k is a single value
        return single_proba(k, subrange) 
    if type(k) == np.ndarray or type(k) == list: 
        proba = []
        for i in range(len(k)):
            proba.append(single_proba(k[i], subrange))
        return np.asarray(proba)



def pmf2(data, q1, q2):
    """Compute the pmf of the secondary quantized dct coefficients based on a primary quantization step.
    
    The first step is to get the indices of the multiples of q1 in the ranges of each c2 in data: that's variable `l`.

    Then we compute the pmf1 with these values.

    When then roughly repeat the process, but this time with only the np.unique(data).
    We then compute the pmf for each value in this 'unique' dataset.

    -----------
    Parameters:

    data -- Vector of dct coefs. Needs to be a numpy array.

    q1 -- Quantization step for the first quantization. 
    
    q2 -- Quantization step for the second Quantization.
    
    """
    rng = get_range(data, q2)

    spread = np.linspace(rng[0], rng[1], rng[1][0] - rng[0][0] + 1)
    quantized = spread / q1
    l = quantized == quantized.astype('int')
    
    q1st = jst.DCTstat(quantized[l].flatten(), q1, x=1, y=1)
    q1st.optimize_parameters()
    pmf1 = q1st.Pv()
    lb1 = q1st.lower_bound

    # `u` means unique
    udata , udx = np.unique(data.astype('int'), True)
    pmf2 = np.zeros(udata[-1] - udata[0] + 1)
    lb2 = np.min(udata)
    uspread = spread[:,udx].astype('int')
    uquantized = quantized[:,udx]
    ul = uquantized == uquantized.astype('int')
    for idx in range(len(udata)):
        pmf2[udata[idx] - lb2] = np.sum(subrange_proba(uspread[ul[:,idx], idx], get_subrange(udata[idx], q2)) * pmf1[uquantized[ul[:,idx], idx].astype('int') - lb1] )
    return pmf2


def set_candidates(data, q2):
    """Identify the candidates as shown in (23), (24) and (26).
    Compute the range for each coef, then check if there's a multiple of the candidate in each of them and for each candidate.
    
    The idea in this function is the following:
    we substract the candidate to the range min and max, and floor the result. If: floor(min/c) - floor(max/c) >= 1,
    it means there is an integer in the range (min/c, max/c), thus ensuring the presence of a candidate's multiple in range(min, max).

    -----------
    Parameters:

    data -- Set of dct coefficients.

    q2 -- Secondary quantization step
    
    -----------
    """

    Rq2 = get_range(data, q2)
    S = []
    N = len(data)
    for cand in range_q1:
        inrange_norm = np.sum((np.floor(Rq2[1] / cand) - np.floor(Rq2[0] / cand)).astype('bool')) / N
        if inrange_norm >= .9:
            S.append(cand)
    return S
    

if __name__=="__main__":


    impath = "data_base/alaska90/00080.jpg"
    

    cstruct = jio.read(impath)
    imgS = cstruct.coef_arrays[0]
    qtb = cstruct.quant_tables[0]
    imgS = jmd.decompress(imgS, 90)
    #imgS = np.array(mpimg.imread(impath))

    qtb2 = jmd.getQmtx(80)
    #repQ2 = np.kron(np.ones((int(imgS.shape[0]/8), int(imgS.shape[1]/8))), qtb2)

    #img = np.round( DCT.bdct(imgS) / repQ2 )#astype('int')
    img = jmd.compress(imgS, 80)
    vct = dct2vec(img)
    """
    repQ = sp.kron(np.ones((int(dct.shape[0]/8), int(dct.shape[1]/8))), qtb)
    img = np.round(DCT.ibdct(dct * repQ)).astype(np.int16)
    """

    data = vct[8].astype(np.int64)
    q2 = qtb2[1][0]
    q1 = qtb[1][0]

    print("Secondary quantization step :", q2)
    print("True primary quantization step :", q1)

    candidates = set_candidates(data, q2)

    print("Set of candidates :\n", candidates)
    
    pmf2e = np.zeros( (len(candidates) , np.max(data) - np.min(data) + 1) ) 
    for idx in range(len(candidates)):
        pmf2e[idx] = pmf2(data, candidates[idx], q2)
    
    color=['r', 'b', 'g', 'c', 'm', 'y', 'k']
    x = range(np.min(data), np.max(data) + 1)
    sp = 200 + 10*np.ceil(len(candidates)/2)+1
    hst = get_hist(data, True)
    
    plt.figure(1)
    for ind in range(pmf2e.shape[0]):
        plt.subplot(sp+ind)
        plt.bar(hst.keys(), hst.values())
        plt.plot(x, pmf2e[ind], color[ind % len(color)])
        plt.legend(str(candidates[ind]))
        #plt.xlim(-20, 21)
    

    kl_distances = np.zeros(len(candidates))
    heights = np.asarray(list(hst.values()))

    not01 = np.asarray(list(hst.keys()))
    not01 = not01[np.abs(not01) >= 2]
    
    for ind in range(len(kl_distances)):
        kl_distances[ind] = kl_distance(pmf2e[ind][not01], heights[not01])
    
    print(kl_distances)
    q1_estimate = candidates[np.argmin(kl_distances)]
    print(q1_estimate)
    plt.show()