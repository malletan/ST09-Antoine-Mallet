import jpegio as jio
import numpy as np
import time
import os
import double_compression as dc
import jpeg_module as jmod


zzorder_x = [0,1,2,1,0,0,1,2]
zzorder_y = [1,0,0,1,2,3,2,1]
zz = np.array([zzorder_x, zzorder_y]).swapaxes(0,1)

x=0
y=1

nbImg = 150


qf = [80, 85, 90, 95, 100]
#dirs = ["alaska80", "alaska85", "alaska90", "alaska95", "alaska100"]
base_dir = "data_base/alaska"



#Count the correct rate of each q2, given the q1, for all the subranges and images.
per_qf1_qf2_correct_rate =  np.zeros((5,4))
#Count the correct rate of each q1, for all q2, subranges and images.
per_qf1_correct_rate = np.zeros((5,)) 
#Count the correct rate for all q1, q2, subranges and images.
total_correct_rate = 0

tstart = time.time()
for i in range(len(qf)):
    qf1 = qf[i]
    print("------> First compression QF:", qf1)
    img_dir = base_dir + str(qf1)
    images = sorted(os.listdir(img_dir))[:nbImg]

    qtb1 = jmod.getQmtx(qf1)
    q1 = qtb1[x][y]

    other_qf = qf.copy()
    del other_qf[i]

    for j in range(len(other_qf)):
        qf2 = other_qf[j]
        print("-> Second compression QF: ", qf2)

        qtb2 = jmod.getQmtx(qf2)
        q2 = qtb2[x][y]

        for img_name in images:
            t0 = time.time()
            cstruct = jio.read(os.path.join(img_dir, img_name))
            dct1 = cstruct.coef_arrays[0]
            img = jmod.decompress(dct1, qf1)
            dct2 = jmod.compress(img, qf2)

            data = jmod.getAC(dct2, x, y).astype('int')
            hst = jmod.get_hist(data, True)
            try:    
                if hst[0]+hst[-1]+hst[1] >=.75:
                    too_compressed += 1
                    #print("Image too compressed. Analysis undetermined.")
                    continue
            except KeyError as e:
                pass

            candidates = dc.set_candidates(data, q2)
            #print("candidates:", candidates)
            pmf2e = np.zeros( (len(candidates) , np.max(data) - np.min(data) + 1) ) 
            for idx in range(len(candidates)):
                pmf2e[idx] = dc.pmf2(data, candidates[idx], q2)

            kl_distances = np.zeros(len(candidates))
            heights = np.asarray(list(hst.values()))

            not01 = np.asarray(list(hst.keys()))    
            not01 = not01[np.abs(not01) >= 2]
            for ind in range(len(kl_distances)):
                kl_distances[ind] = dc.kl_distance(pmf2e[ind][not01], heights[not01])

            q1_estimate = candidates[np.argmin(kl_distances)]

            print(">{0} processed in {1}. Test: {2}, Truth: {3}<".format(img_name, time.time() - t0, q1_estimate, q1))


            if q1_estimate == q1:
                per_qf1_qf2_correct_rate[i][j] += 1 

    print("Correct guess rate:", per_qf1_qf2_correct_rate[i])   

per_qf1_qf2_correct_rate /= nbImg
print("Results Summary:\n", per_qf1_qf2_correct_rate)
print("Computation time:", time.time() - tstart)
