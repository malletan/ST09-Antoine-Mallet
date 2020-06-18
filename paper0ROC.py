import numpy as np
import jpegio as jpio
import jstegclass as jsteg
import jpeg_stats as stats
import matplotlib.pyplot as plt
import os
import glob
#from jpeg import jpeg
import multiprocessing
import time

#import warnings
#warnings.filterwarnings('ignore')

xlim = [ 0 , 1 , 2, 3, 4, 5, 6, 7]
ylim = xlim

RmCsteBlock = True # False
if RmCsteBlock:
    print("--> REMOVING ALL CONSTANT BLOCKS <--")
else:
    print("--> CONSTANT BLOCKS are kept while the statistical model be by badly off ! <--")


def dct2vec(DCT):
    nbDCT = DCT.shape
    dctVec = np.zeros( [ 64 , int(nbDCT[0]*nbDCT[1]/64) ] )
    for x in xlim:
        for y in ylim:
            if x != 0 or y != 0:
                dctVec[x*8+y,:] = DCT[x::8,y::8].flatten()
    """Here we remove constant blocks, for both lowering computational complexity and increasing model accuracy (you can comment those lines in order no to do so)"""
    if RmCsteBlock:
        blkMin = np.min(dctVec,0)
        blkMax = np.max(dctVec,0)
        idxCsteBlk = (blkMin != blkMax)
        dctVec = dctVec[:,idxCsteBlk]
    return dctVec



payload = 0.1
images = '/home/cogrannr/Recherche/ImgDatasets/BOSS_jpeg75_cover/'
images = "data_base/alaska85/"
output = "statistics/ROC/"

def compute_LR( impath ):
    c_struct=jpio.read(impath)
#    c_struct=jpeg(impath)
    cover = c_struct.coef_arrays[0]
    qtb = c_struct.quant_tables[0]

    coverDct = dct2vec(cover)
    stegoDct = jsteg.JSteg_Embedding(coverDct, payload)

    stat_cover = jsteg.IMGstat(coverDct, qtb , x = xlim , y = ylim , payload = payload)
    stat_cover.log_ratio_test()
    glr_h0 = stat_cover.LRTnorm

    stat_stego = jsteg.IMGstat(stegoDct, qtb , x = xlim , y = ylim , payload = payload)
    stat_stego.log_ratio_test()
    glr_h1 = stat_stego.LRTnorm

    return glr_h0 , glr_h1



# DCT mode we'll use to draw the ROC curve

if not os.path.exists(images):
    print("The path to the images does not exist.")
else:
    if not os.path.exists(output):
        print("Creating Reciever-Operator-Curve directory..")
        os.makedirs(output)
    else:
        print("Reciever-Operator-Curve already exists.")

#    sorted_img = sorted(os.listdir(images))
    sorted_img = sorted(glob.glob(images + '*.jpg' ))
    
    nbImg = 240
    sorted_img = sorted_img[:nbImg]

    glr_h0 = []
    glr_h1 = []
    """
    t = time.time()
    nbCores = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(nbCores)
    for LRh0 ,  LRh1 in pool.map(compute_LR , sorted_img):
        glr_h0.append(LRh0)
        glr_h1.append(LRh1)
    pool.close()
    pool.join()
    print('--> Elapsed Time : ' , time.time() - t , ' \t Time per image : ' , (time.time() - t)/len(sorted_img) , ' <--')
    """

    tStart = time.time()
    for impath in sorted_img: 
        tBegin = time.time()
        [ tmp_RVG0 , tmp_RVG1 ] = compute_LR(impath)
        glr_h0.append(tmp_RVG0)
        glr_h1.append(tmp_RVG1)
        print( ' -->  Image Name = ' , impath , ' \t LR0 = %.4f  -- LR1 = %.3f \t --> Elapsed Time : %.3f -- Total Time = %.3f <-- ' % ( tmp_RVG0 , tmp_RVG1 , time.time() - tBegin , time.time() - tStart ) )

    print(' ***** NB nan = ' , np.sum( np.isnan( glr_h1 ) ) , ' / ' , nbImg , '  ***** ')

    pfa, power = jsteg.getROC(glr_h0, glr_h1)

    plt.figure(num="ROC_100")
    plt.plot(pfa, power)
    plt.axis([0, 1, 0.0, 1])
    plt.xlabel(r'$\alpha_0$ --- false positive rate', fontsize=12)
    plt.ylabel(r'$\beta_1$ --- detection rate', fontsize=12)
    plt.savefig(output + "ROC_100")


    plt.figure(num="ROClog_100")
    plt.semilogx(pfa, power)
    plt.axis([0, 1, 0.0, 1])
    plt.xlabel(r'$\alpha_0$ --- false positive rate', fontsize=12)
    plt.ylabel(r'$\beta_1$ --- detection rate', fontsize=12)
    plt.show()
    plt.savefig(output + 'ROClog_100' )


