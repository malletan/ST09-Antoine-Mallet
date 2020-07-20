import jpegio as jio
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import double_compression as dc
import jpeg_module as jmod
import dct
import os
import time



impath = "data_base/boss_rnd128/BOSS_RndQF128"
q2 = 5
qtb2 = np.ones((8,8)) * q2


nbImg = 100


# Setting all the quantization steps for subrange (0,1) for qf from 50 to 100
q01 = np.zeros(51)
for q in range(len(q01)):
    q01[q] = jmod.getQmtx(q+50)[0][1]


correct_estimation = 0
q1_overbound = 0
q1_divisor = 0
too_compressed = 0
counter = 0

tstart = time.time()
for img_name in os.listdir(impath)[:nbImg]:
    t0 = time.time()
    
    q1 = q01[int(img_name[-6:-4]) - 50] # To get the QF in the name
    if q1 > 20:
        q1_overbound += 1
        #print("First quantization too high:", q1)
        continue

    if q2 % q1 == 0:
        q1_divisor += 1
        print("First quantization step is a divisor of the second one:\nq1 = {0}\tq2 = {1}".format(q1, q2))
        continue


    #print("\n")
    #img = np.array(mpimg.imread(os.path.join(impath, img_name)))

    cstruct = jio.read(os.path.join(impath, img_name))
    img = cstruct.coef_arrays[0]
    qtb = cstruct.quant_tables[0]
    img = jmod.decompress(img, 80)
    repQ2 = jmod.repmat(qtb2, (int(img.shape[0]/8), int(img.shape[1]/8)))

    img = np.round( dct.bdct(img) / repQ2 )
    data = jmod.dct2vec(img)[1].astype('int') # subband (0,1)
    hst = jmod.get_hist(data, True)

    try:    
        if hst[0]+hst[-1]+hst[1] >=.75:
            too_compressed += 1
            print("Image too compressed. Analysis undetermined.")
            continue
    except KeyError as e:
        pass



    candidates = dc.set_candidates(data, q2)
    #print("True q1 in set of candidates:", q1 in candidates)
    print("candidates:", candidates)
    pmf2e = np.zeros( (len(candidates) , np.max(data) - np.min(data) + 1) ) 
    for idx in range(len(candidates)):
        pmf2e[idx] = dc.pmf2(data, candidates[idx], q2)



    """
    color=['r', 'b', 'g', 'c', 'm', 'y', 'k']
    x = range(np.min(data), np.max(data) + 1)
    sp = 200 + 10*np.ceil(len(candidates)/2)+1
    hst = dc.get_hist(data, True)
    
    plt.figure(1)
    for ind in range(pmf2e.shape[0]):
        plt.subplot(sp+ind)
        plt.bar(hst.keys(), hst.values())
        plt.plot(x, pmf2e[ind], color[ind % len(color)])
        plt.legend(str(candidates[ind]))

    """
    
    kl_distances = np.zeros(len(candidates))
    heights = np.asarray(list(hst.values()))

    not01 = np.asarray(list(hst.keys()))
    not01 = not01[np.abs(not01) >= 2]
    for ind in range(len(kl_distances)):
        kl_distances[ind] = dc.kl_distance(pmf2e[ind][not01], heights[not01])

    q1_estimate = candidates[np.argmin(kl_distances)]

    t1 = time.time()

    if q1_estimate == q1:
        correct_estimation += 1
    
    print("--> {0}: q1 = {1}, estimate = {2}. Done in {3} <--".format(img_name, q1, q1_estimate, t1 - t0))
    counter += 1


print("===> Correct estimation rate: {0} <===".format(correct_estimation / nbImg))
print("Analysis conducted over {0} images".format(counter))
print("q1 over 20 count:", q1_overbound)
print("q1 divisor of q2:", q1_divisor)
print("data too compressed:", too_compressed)
