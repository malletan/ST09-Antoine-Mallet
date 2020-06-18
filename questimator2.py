import jpegio as jio
import numpy as np
import matplotlib.pyplot as plt
import jpeg_stats as jst
import dct
import jpeg_module as jmd
import time


xlim = [ 0 , 1 , 2, 3, 4, 5, 6, 7]
ylim = xlim


RmCsteBlock = True # False
if RmCsteBlock:
    print("--> REMOVING ALL CONSTANT BLOCKS <--")
else:
    print("--> CONSTANT BLOCKS are kept while the statistical model be by badly off ! <--")


def validity(blocks):
    valid = []
    for b in blocks:
        valid.append( not(np.max(b) == np.min(b) or np.any(b <= 3) or np.any(b >= 252)) )

    return valid

def im2dctvec(IMG):
    dim = IMG.shape
    dctvec = []
    blocks = IMG.reshape((int(dim[0] * dim[1] / 64), 8, 8))
    print(blocks.shape)
    if RmCsteBlock:    
        blocks = blocks[validity(blocks)]
        print(blocks.shape)
    for b in blocks:
        b = dct.ibdct(b - 128).astype('int')
        dctvec.append(b.flatten())

    return np.swapaxes(np.asarray(dctvec), 0, 1)



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
        print("DCT before removing:", len(dctVec), len(dctVec[0]))
        dctVec = dctVec[:,idxCsteBlk]
        print("DCT after removing:", len(dctVec), len(dctVec[0]))
    return dctVec


def compute_param(imgVec):
    stats = [] # Will list the dct stat object for each subband
    params = [] #Will hold the parameters for each subband
    for subband, i in zip( imgVec, range(len(imgVec)) ):
        x, y = np.unravel_index(i, (8,8))
        stats.append(jst.DCTstat(subband, 1, x=x, y=y))
        stats[-1].compute()
        stats[-1].optimize_parameters()
        params.append( [stats[-1].eta, stats[-1].nu] )

    return np.swapaxes(np.asarray(params), 0, 1)


def guess_q(subband):

    hist = get_hist(subband) # The function doesn't seem too inferior speed-wise compared to numpy's built-in `histogram` function
    #print(hist)
    
    plt.figure(2)
    plt.bar(hist.keys(), hist.values())
    #maxQ = np.argmax(hist[1][np.argmax(hist[0])+1:])
    maxval = 0
    maxQ = 0
    for x in hist:
    	if x and hist[x] >= maxval and not x:  
    		initial_Q = x
    		maxval = hist[x]
        
    

    print("Initial guess: ", maxQ)
    candidates = []
    for i in range(2, int(maxQ / 2 + 1)): 
        if not initial_Q % i:
            print(i, " divises ", maxQ)
            candidates.append(i)
    candidates.append(maxQ)

    
    return candidates


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
        print("Percentage of values in range: ", percent)
    
    #hist = np.fromiter(tmp.items(), dtype=dict(names=['id', 'data'], formats=['V16', 'V16']), count=len(tmp))
    return tmp


#img = plt.imread("data_base/alaska90/00004.jpg")

cstruct = jio.read("data_base/alaska90/00004.jpg")
img = cstruct.coef_arrays[0]

print(np.max(img), np.min(img))
plt.figure(1)
plt.imshow(img, cmap='gray')

print(img.shape)
DCT = np.round(dct.ibdct(img-128))

qtb = jmd.getQmtx(90)

t0 = time.time()
#dctvec = dct2vec(DCT)
dctvec = im2dctvec(img)
print("DCT transform and subband splitting in ", time.time() - t0)

#params = compute_param(dctvec)
N = 1 # The subband of which we try to determine the quantization step 


candy = guess_q(dctvec[N])
print("Candidates: ", candy)
print("True Quantization step: ", qtb[0][1])
plt.show()