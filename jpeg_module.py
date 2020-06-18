from PIL import Image
import numpy as np
import dct


def crop8x8(img, size):
    """Make an image shape multiple of (8x8)"""
    return img[:img.shape[0] - img.shape[0] % size, :img.shape[1] - img.shape[1] % size]


def load_image(filename):
    """Open an image and return it as an numpy array"""
    img = Image.open(filename)
    img.load()
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def YCbCr2RGB(img):
    """Take an image in YCbCr domain and return it in RGB domain"""
    if len(img.shape) is not 3:
        print("Incorrect input image format: {0} dimensions instead of expected {1}".format(len(img.shape), 3))
    X = np.array([[1, 0, 1.402], [1, -.34414, -0.71414], [1, 1.772, 0]])
    rgb = img.astype(np.float)
    rgb[:,:,[1,2]] -=128
    rgb = rgb.dot(np.transpose(X))
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def RGB2YCbCr(img):
    """Take an image in YCbCr domain and return it in RGB domain"""
    if len(img.shape) is not 3:
        print("Incorrect input image format: {0} dimensions instead of expected {1}".format(len(img.shape), 3))
        return img
    X = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = img.dot(X.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def RGB2GRAY(img):
    if len(img.shape) is not 3:
        print("Incorrect input image format: {0} dimensions instead of expected {1}".format(len(img.shape), 3))
        return img
    X = np.array([0.3, 0.59, 0.11])
    return img.dot(X.T)


def getQmtx(quality, channel='Y'):
    """Computes and returns the quantization matrix according to the input quality. By default, it will compute on the Y channel array Qy,
       as it is used for both Y channel in multi channel (YCbCr) images and for mono channel (grayscaled) images."""
    Qy = np.array( # Quantization table for luminance
    [
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ])
    Qc = np.array( # Quantization table for Crominance channels 
    [
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
    ])  
    if quality <= 0: quality = 1
    if quality > 100: quality = 100
    if quality < 50: qualityFactor = 5000 / quality 
    if quality >=50: qualityFactor = 200 - quality*2

    if channel is not getQmtx.__defaults__[0]: # if the input is not Y or grayscale channel
        quantMatrix = np.floor( (Qc * qualityFactor + 50) /100 )
    else:
        quantMatrix = np.floor( (Qy * qualityFactor + 50) /100 )

    quantMatrix[quantMatrix<1] = 1 

    return quantMatrix


def repmat(Q, rep): return np.kron(np.ones(rep), Q)

def quantise(C, Q):
    [k,l] = C.shape
    [m,n] = Q.shape
    rep = (int(k/m), int(l/n))
    #rep = (k/m, l/n)
    return   C / repmat(Q, rep) 


def dequantise(C, Q):
    [k,l] = C.shape
    [m,n] = Q.shape
    rep = (int(k/m), int(l/n))
    #rep = (k/m, l/n)
    return  C * repmat(Q, rep)


def getFreq(channel, quality_factor, ctype='Y'):
    Qmtx = getQmtx(quality_factor, ctype)
    return np.around( quantise( dct.bdct(channel), Qmtx ) )

def getSpatial(channel, quality_factor, ctype='Y'):
    Qmtx = getQmtx(quality_factor, ctype)
    return dequantise( dct.ibdct(channel), Qmtx )


def compress(img, quality_factor):
    """Compresses images that are either in GRAYSCALE or YCbCr color modes.
       In YCbCr mode, returns an array of each transformed channel separated. 
       Note: The processing of multi-channel is not robust. """
    if len(img.shape) < 3:
        return getFreq(img - 128, quality_factor)
    elif img.shape[2] == 3: 
        c0 = img[:,:,0] #- 128
        c1 = img[:,:,1] #- 128
        c2 = img[:,:,2] #- 128
        freqs = [ 
            getFreq(c0, quality_factor),
            getFreq(c1, quality_factor, ctype='Cb'),
            getFreq(c2, quality_factor, ctype='Cr'),
        ]
        return np.asarray(freqs)
    else:
        print("Invalid dimensions for the image: 1 or 3 color channel(s) expected.")

def decompress(freqs, quality_factor):
    if len(freqs.shape) < 3:
        return getSpatial(freqs, quality_factor) + 128
    elif freqs.shape[0] == 3: 
        img = np.ndarray( (freqs.shape[1], freqs.shape[2], freqs.shape[0]) ) # We change the dimensions back to the original ones
        for i in range(freqs.shape[0]):
            img[:,:,i] = getSpatial(freqs[i], quality_factor) 
        return img
    else:
        print("Invalid dimensions for the image: 1 or 3 color channel(s) expected.")
