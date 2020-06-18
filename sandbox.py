import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
import dct
from scipy import kron
from scipy.signal import argrelextrema
import jpeg_stats as jstat

RmCsteBlock = True
if RmCsteBlock:
    print("--> REMOVING ALL CONSTANT BLOCKS <--")
else:
    print("--> CONSTANT BLOCKS are kept while the statistical model be by badly off ! <--")
xlim = [0, 1, 2, 3, 4, 5, 6, 7]
ylim = xlim

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
        percent = itt / ott
        #print("Percentage of values in range: ", percent)
    
    return tmp


def compute_param(imgVec):
    stats = [] # Will list the dct stat object for each subband
    params = [] #Will hold the parameters for each subband
    for subband, i in zip( imgVec, range(len(imgVec)) ):
        x, y = np.unravel_index(i, (8,8))
        stats.append(jstat.DCTstat(subband, 1, x=x, y=y))
        stats[-1].compute()
        stats[-1].optimize_parameters()
        params.append( [stats[-1].eta, stats[-1].nu] )

    return np.swapaxes(np.asarray(params), 0, 1)


def divisors(n):
    """Return all divisors of n."""
    for i in range(1, int((n+1)/2)):
        if not n % i : yield i
    yield n



def iqfRatio(Q, data):
    """Return the ratio of integer quantized forward of an list"""
    quantized = np.array(data) / Q
    quantized_rounded = quantized.astype('int')
    return np.count_nonzero((quantized - quantized_rounded) == 0) / len(data)






c_struct = jio.read("data_base/alaska80/00004.jpg")
img = c_struct.coef_arrays[0]
qtb = c_struct.quant_tables[0].flatten()

vectorized = dct2vec(img)
#params = compute_param(vectorized)

Qguess = np.zeros(64)

iqfThreshold = .2


for nsb in range(64):

    print(nsb)
    #print("Number of subband : ", len(vectorized))
    data = vectorized[nsb]

    non_zero = np.count_nonzero(data)
    if non_zero < len(data) / 10:
        print("Subband too compressed: {0}percent of the coefficients are non zero.".format(non_zero/len(data)*100))
        continue
        
    #Here we do the decompression and the forward dct back again, in order to study an uncompressed-like image
    backward_img = np.round(dct.ibdct(img*kron(np.ones((int(img.shape[0]/8), int(img.shape[1]/8))), qtb.reshape((8,8))))).astype('int')
    recompressed = np.round(dct.bdct(backward_img)).astype('int')

    recompressed = np.round(dct.bdct(np.round(dct.ibdct(img*kron(np.ones((int(img.shape[0]/8), int(img.shape[1]/8))), qtb.reshape((8,8))))).astype('int'))).astype('int')

    data = dct2vec(recompressed)[nsb]
    hist = get_hist(data)

    keys = np.asarray(list(hist.keys()))
    values = np.asarray(list(hist.values()))

    poskeys = keys[keys > 0]
    posvals = values[keys > 0]

    local_maxima_indices = argrelextrema(posvals, np.greater)[0]
    #print("All the local maxima: ", local_maxima_indices)

    local_maxima_values = posvals[local_maxima_indices]
    #print("Local maxima values: ", local_maxima_values)
    if not local_maxima_values.size:
        print("No maximum found.")
        print("local_maxima_indices: ", local_maxima_indices)
        print("local_maxima_values: ", local_maxima_values)
        continue

    max_of_maxima = local_maxima_indices[np.argmax(local_maxima_values)]
    #print("Index where the highest value is met: ", max_of_maxima)

    Qcandidate = poskeys[max_of_maxima]
    #print("The candidate is :", Qcandidate)

    divisors_of_candidate = divisors(Qcandidate)
    candidates = []
    for d in divisors_of_candidate:
        if iqfRatio(d, data) > iqfThreshold:
            candidates.append(d)

    print("The candidates are:", candidates)
    subband_stat = jstat.DCTstat(data, 1, int(nsb/8), nsb%8)
    bestfit = subband_stat.optimize_quant_step(candidates)
    #bestfit = subband_stat.optimize_quant_step(candidates)


    Qguess[nsb] = bestfit


print("Real quantization table:\n", qtb.reshape((8,8)))
print("Guessed quantization table:\n", Qguess.reshape((8,8)))

non_guessed = 0
guessed_right = 0
guessed_wrong = 0

for guessed, true in zip(Qguess, qtb):
    if not guessed:
        non_guessed +=1
    elif guessed == true:
        guessed_right += 1
    elif guessed != true:
        guessed_wrong += 1

print(guessed_right, " Step correctly guessed.")
print(guessed_wrong, " Step incorrectly guessed.")
print(non_guessed, "Step avoided")

print("Out of {0} guesses, {1} % correct".format(guessed_right + guessed_wrong, guessed_right/(guessed_right+guessed_wrong)*100))


"""
        
print("Length of each subband : ", len(data))
print("Delta in the subband: ", min(data), max(data))

hist_height = get_hist(data)

plt.figure(1)
plt.bar(hist_height.keys(), hist_height.values())
plt.axis([-50, 50, 0, max(hist_height.values())])



backward_img = np.round(dct.ibdct(img*kron(np.ones((int(img.shape[0]/8), int(img.shape[1]/8))), qtb.reshape((8,8))))).astype('int')

recompressed = np.round(dct.bdct(backward_img)).astype('int')


data2 = dct2vec(recompressed)[nsb]
hist2 = get_hist(data2)
plt.figure(2)
plt.title("True quantization step: {0}".format(qtb[nsb]))
plt.bar(hist2.keys(), hist2.values())
plt.axis([-40, 40, 0, 2*max(hist2.values())])


#We take the positive half of the hist, since Q > 0
#keys = list(hist2.keys())
#values = list(hist2.values())
keys = np.asarray(list(hist2.keys()))
values = np.asarray(list(hist2.values()))

print(keys)
print(values)
#arrayhist = np.swapaxes(np.asarray([keys, values]), 0, 1)
arrayhist = np.asarray([keys, values])
print(arrayhist)
poskeys = keys[keys > 0]
posvals = values[keys > 0]
print(poskeys)
print(posvals)

local_maxima_indices = argrelextrema(posvals, np.greater)[0]
print("All the local maxima: ", local_maxima_indices)

local_maxima_values = posvals[local_maxima_indices]
print("Local maxima values: ", local_maxima_values)
max_of_maxima = local_maxima_indices[np.argmax(local_maxima_values)]
print("Index where the highest value is met: ", max_of_maxima)

Qcandidate = poskeys[max_of_maxima]
print("The candidate is :", Qcandidate)


plt.show()
"""