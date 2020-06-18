import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
from scipy import kron
from scipy.signal import argrelextrema

import jstegclass as jsteg
import jpeg_stats as jstat
import dct

import time
import glob
import os


xlim = [ 0 , 1 , 2, 3, 4, 5, 6, 7]
ylim = xlim

RmCsteBlock = True # False
if RmCsteBlock:
    print("--> REMOVING ALL CONSTANT BLOCKS <--")
else:
    print("--> CONSTANT BLOCKS are kept while the statistical model be by badly off ! <--")


def dct2vec(DCT):
    nbdct = DCT.shape
    dctVec = np.zeros( [ 64 , int(nbdct[0]*nbdct[1]/64) ] )
    for x in xlim:
        for y in ylim:
            if x != 0 or y != 0:
                dctVec[x*8+y,:] = DCT[x::8,y::8].flatten()
    """Here we remove constant blocks, for both lowering computational complexity and increasing model accuracy (you can comment those lines in order no to do so)"""
    if RmCsteBlock:
        blkMin = np.min(dctVec[1::],0) # [1::] because if the block is constant, its dct will be constant for the AC coefficients
        blkMax = np.max(dctVec[1::],0) # same 
        idxCsteBlk = (blkMin != blkMax)
        dctVec = dctVec[:,idxCsteBlk]
        #print("length after removing:", len(dctVec), len(dctVec[0]))
    return dctVec


def  get_hist(data, norm=False):
    """Creates a dictionary which keys are the histogram's bins and """
    tmp = {}
    minval = int(min(data))
    maxval = int(max(data))
    itt = 0 # total count within the bins
    ott = 0 # total count of all the vals 
    #Initialize the dictionary
    for b in range(minval, maxval):
        tmp[b] = 0
    #Filling the values
    for d in data:
        ott += 1 # we count even the values outside of the range
        if d > minval and d < maxval:
            itt += 1
            tmp[d] += 1
    
    if norm is True:
        for v in tmp:
            tmp[v] /= ott
        #percent = itt / ott
        #print("Percentage of values in range: ", percent)
    
    return tmp


def divisors(n):
    """Return all divisors of n."""
    for i in range(1, int((n+1)/2)):
        if not n % i : yield i
    yield n



def iqfRatio(Q, data):
    """Return the ratio of integer quantized forward of a list"""
    quantized = np.array(data) / Q
    quantized_rounded = quantized.astype('int')
    return np.count_nonzero((quantized - quantized_rounded) == 0) / len(data)



def Qguess(vectorized_img):
    bestfits = np.zeros(64)
    for nsb in range(64):
        data = vectorized_img[nsb]

        non_zero = np.count_nonzero(data)
        if non_zero < len(data) / 10:
            continue    

        hist = get_hist(data)
        keys = np.asarray(list(hist.keys()))
        values = np.asarray(list(hist.values()))
        poskeys = keys[keys > 0]
        posvals = values[keys > 0]
        
        local_maxima_indices = argrelextrema(posvals, np.greater)[0]
        local_maxima_values = posvals[local_maxima_indices]

        if not local_maxima_values.size:
            continue

        max_of_maxima = local_maxima_indices[np.argmax(local_maxima_values)]

        Qcandidate = poskeys[max_of_maxima]        

        """
        divisors_of_candidate = divisors(Qcandidate)
        candidates = []
        for d in divisors_of_candidate:
            if iqfRatio(d, data) > iqfThreshold:
                candidates.append(d)

        #print("The candidates are:", candidates)
        subband_stat = jstat.DCTstat(data, 1, int(nsb/8), nsb%8)
        bestfits[nsb] = subband_stat.optimize_quant_step(candidates)
        """
        bestfits[nsb] = Qcandidate

    return bestfits




images = "data_base/alaska100/"
nbImg = 240
iqfThreshold = .2

if not os.path.exists(images):
    print("The path to the images does not exist.")



sorted_img = sorted(glob.glob(images + '*.jpg' ))[:nbImg]

guess_rate = 0
correct_rate = 0

tstart = time.time()
for impath in sorted_img:
    t0 = time.time()
    cstruct = jio.read(impath)
    img = cstruct.coef_arrays[0]
    qtb = cstruct.quant_tables[0].flatten()

    img = np.round(dct.bdct(np.round(dct.ibdct(img*kron(np.ones((int(img.shape[0]/8), int(img.shape[1]/8))), qtb.reshape((8,8))))).astype('int'))).astype('int')

    vec = dct2vec(img)

    q_guessed = Qguess(vec)

    non_guessed = 0
    guessed_right = 0
    guessed_wrong = 0

    for guessed, true in zip(q_guessed, qtb):
        if not guessed:
            non_guessed +=1
        elif guessed == true:
            guessed_right += 1
        elif guessed != true:
            guessed_wrong += 1
    total_guess_percentage = (guessed_right + guessed_wrong) / .64
    correct_guess_percentage = guessed_right / (guessed_right + guessed_wrong) * 100

    print('--> Image = ', impath, 'guessing rate = %.3f\t correct rate = %.3f\t --> Elapsed time = %.3f\t -- Total = %.3f' % (total_guess_percentage, correct_guess_percentage, time.time() - t0, time.time() - tstart))

    guess_rate += total_guess_percentage
    correct_rate += correct_guess_percentage

guess_rate /= nbImg
correct_rate /= nbImg
print("==> RESULT FOR IMAGES IN {0} <==".format(images))
print("--> Guessing rate =  {0}".format(guess_rate))
print("--> Correct guess in {0} percent of cases".format(correct_rate))

