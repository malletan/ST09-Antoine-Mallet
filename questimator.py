import numpy as np
import matplotlib.pyplot as plt
import jpegio as jio
import jpeg_stats as jpeg
import dct
import jpeg_module as jmod
import jpeg_stats as stats






def is_valid(block):
    """
       Check wheter a block is valid for computation.
       An invalid block is either:
        - uniform: all the values in the block are identical
        - saturated: the minval is 0 or the maxval is 255
    """
    # Check for uniformity
    #if np.amin(block) == np.amax(block) or np.any(block == 0) or np.any(block == 255): return False
    if np.amin(block) == np.amax(block): 
        print("Pas valide")
        return False
    """
    nbrow = block.shape[0]
    nbcol = block.shape[1]
    for x in range(nbrow):
        for y in range(nbcol):
            #Check for saturation
            if block[x][y] == 0 or block[x][y] == 255:
                print("Block saturated at ({0},{1})".format(x, y))
                return False
    #At this point, we know the block is valid for computation
    #print("Block is valid")
    """
    return True

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
"""
def  get_hist(data, bins, norm=False):
    hist = {}
    minval = int(min(bins))
    maxval = int(max(bins))
    itt = 0 # total count within the bins
    ott = 0 # total count of all the vals 
    #Initialize the dictionary
    for b in bins:
        hist[b] = 0
    #Filling the values
    for d in data:
        ott += 1 # we count even the values outside of the range
        if d > minval and d < maxval:
            itt += 1
            hist[d] += 1
    
    if norm is True:
        for v in hist:
            hist[v] /= ott
        percent = itt / ott
        print("Percentage of values in range: ", percent)
    
    return hist
"""

def max_bin(hist):
	"""Returns the non-zero bin with the highest value."""
	maxval = 0
	maxkey = 0
	for key in hist:
		if key and hist[key] >= maxval:  
			maxkey = key
			maxval = hist[key]
	return maxkey


def im2dctvec(img):
    """
        Checks the validity of each block, and computes the dct and rounds the coefficients if the block is valid.
        Then stores the coefficients in a 64 lists' list.
    """
    #vec_dct = [ [] for n_of_subband in range(64)]
    vec_dct = []
    nbrow = img.shape[0]
    nbcol = img.shape[1]
    for blck_x in range(0, nbrow, 8):
        for blck_y in range(0, nbcol, 8):
            block  = img[blck_x:blck_x + 8, blck_y:blck_y + 8]
            if is_valid(block):
                #print("Valide")
                flattened = dct.bdct(block).astype('int').flatten()
                vec_dct.append(flattened)
    print(np.asarray(vec_dct).shape)
    
    return np.swapaxes(np.asarray(vec_dct), 0, 1)


def is_iqf(val):
    """
        Test if the parameter val is an integer.
    """
    return (np.round(val) == int(val))


def n_of_iqf(subband, q):
    """
        Gets the number of Integer Quantized Forward coefficients from a sub-band, given an estimated quantization step.
    """
    n_q = 0
    for c in subband:
        if is_iqf(c / q): n_q += 1
    return n_q / len(subband)















#Loading the compressed file
c_struct = jio.read("data_base/alaska95/00002.jpg") # From jpeg file
img = c_struct.coef_arrays[0]
#img = np.load("data_base/img/80/00002.npy").astype(np.int32) # From npy file

n = 9

#Fetching the sub-bdand
Vk = im2dctvec(img)

#Loading the quantization table
qtb = c_struct.quant_tables[0]
#qtb = jmod.getQmtx(80)

print(qtb)

#Fetching the true quantization step
ind = np.unravel_index(n, (8,8))
true_q = qtb[ind[0]][ind[1]]
subband = Vk[n]


iqf_ratios = [ 1 ]
for q in range(1, 30):
    iqf_ratio = n_of_iqf(subband, q)
    iqf_ratios.append(iqf_ratio)

unq_stat = stats.DCTstat(Vk[n], 1, ind[0], ind[1]) # We can deal with the unquantized coefficient as if they were quantized ones with a step = 1
unq_stat.compute()
print("Unquantized eta: ", unq_stat.eta, "\tUnquantized nu: ", unq_stat.nu)

optim = neldm.ParamOptimizer(unq_stat)
print(optim.opt)
#print(unq_stat.etaML, unq_stat.nuML)






#lin = range(np.amin(Vk[8]), np.amax(Vk[8]))
lin =range(-35, 35)

hst = get_hist(Vk[n])

plt.figure(1)
#plt.hist(Vk[n], bins=lin, density=True)
plt.bar(hst.keys(), hst.values(), color='b')
plt.legend(["True quantization step:"+str(true_q)])
plt.figure(2)
plt.plot(range(30), iqf_ratios, '-r')
plt.xticks(range(30))
plt.yticks(np.linspace(min(iqf_ratios), max(iqf_ratios), num=10))
plt.grid()
plt.legend(["True quantization step:"+str(true_q)])
plt.show()


"""
i = 0
nbrow = img.shape[0]
nbcol = img.shape[1]
for blckx in range(0, nbrow, 8):
    for blcky in range(0, nbcol, 8):
        block  = img[blckx:blckx + 8, blcky:blcky + 8]
        if not is_valid(block):
            print("Block ({0},{1}) not valid".format(blckx, blcky))
            i += 1

print("Number of invalid blocks: ", i)
print("Percentage of valid blocks: ", 1 - i / (248*248/64))
"""