import jpeg_stats as stats
import numpy as np
import os 
import matplotlib.pyplot as plt
import NelderMead as nm

images = "data_base/images/"
qtables = "data_base/qtables/"
output = "statistics/"

quality_factor = 80
number_of_images = 2 

im_dir = os.path.join(images, str(quality_factor))
q_dir = os.path.join(qtables, str(quality_factor))

sorted_images = sorted(os.listdir(im_dir))[:number_of_images]
sorted_qtables = sorted(os.listdir(q_dir))[:number_of_images]

output_dir = os.path.join(output, str(quality_factor))
if not os.path.exists(output_dir):
    print("{0} does not exist. It will be created..".format(output_dir))
    os.makedirs(output_dir)
else:
    print("{0} already exists..".format(output_dir))


for idx in range(len(sorted_images)):
    dct = np.load(os.path.join(im_dir, sorted_images[idx]))[0]
    Qmtx = np.load(os.path.join(q_dir, sorted_qtables[idx]))[0]

    im_name = os.path.splitext(sorted_images[idx])[0]

    validity = np.ndarray((8,8), dtype=np.int8) #Check wether the parameters are valid or not
    validity[0][0] = True # dct mode (0,0) is not handled


    for x in range(8):
        for y in range(8):
            if x != 0 or y != 0:
                print(x, y)
                stat_name = "{0}_({1}-{2})".format(im_name, x, y)
                dct_stat = stats.DCTstat(dct, Qmtx, x, y)
                dct_stat.compute()
                optimizer = nm.ParamOptimizer(dct_stat)

                #Checks the validity (greater or equal to zero) of each set of parameter. valid=True, invalid=False
                validity[x][y] = int((dct_stat.eta >= 0) and (dct_stat.nu >= 0))

                #Constructing the figure that shows the histogram, probability density function and probability mass function of the dataset
                plt.figure(num=stat_name)
                PMF=dct_stat.PvRC() #should either change to Pv() or check PvRC at the bottom of the jpeg_stats.py script
                plt.hist(dct_stat.data, bins=dct_stat.L  , density=True, label="eta = "+str(round(dct_stat.eta, 3))+"  nu = "+str(round(dct_stat.nu, 3))+"  Q = "+str(dct_stat.Q))
                #plt.plot(dct_stat.X , dct_stat.Fi, label='PDF')
                X = dct_stat.L
                plt.plot(X, PMF, 'r+', label='PMF')
                PMFoptim = dct_stat.PvRC(e=optimizer.opt.x[0] , n=optimizer.opt.x[1]) #should either change to Pv() or check PvRC at the bottom of the jpeg_stats.py script
                plt.plot(X, PMFoptim, 'g+', label='PMFoptim')
                plt.legend(loc='upper left')

                savePath = os.path.join(output_dir, stat_name+'.png')
                plt.savefig(savePath)
                plt.close()

    unique, counts = np.unique(validity, return_counts=True)
    dic = dict(zip(unique, counts))
    percent = dic[1] / 64
    print( im_name + " : Valid Parameter Percentage:" + str(percent) )