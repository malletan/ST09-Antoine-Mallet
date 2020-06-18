from numpy import dot, linalg
import numpy as np

def auxcos(x,u):
    return np.cos((np.pi/8) * (x+.5) * u)

def cosmat(M=8,N=8):
    C = np.array([ [ auxcos(x,u) for u in range(N) ] for x in range(M) ])
    C[:,0] = C[:,0] / np.sqrt(2)
    return C

auxM = cosmat(8,8)
invM = linalg.inv(auxM)
auxT = np.transpose(auxM)
invT = np.transpose(invM)

def dct2(g):
    assert (8,8) == np.shape(g)
    return dot(auxT, dot(g, auxM))

def idct2(g):
    assert(8,8) == np.shape(g)
    return dot(invT, dot(g, invM))

def bdct(C, f=dct2):
    (M,N) = np.shape(C)
    assert M%8 == 0
    assert N%8 == 0
    S = np.ndarray((M,N))
    for i in range(0, M, 8):
        for j in range(0, N, 8):
            S[i:i+8, j:j+8] = f( C[i:i+8, j:j+8] )
    return S

def ibdct(C): return bdct(C, f=idct2)
