import numpy as np
import matplotlib.pyplot as plt
import os
import jpeg_stats as stats
import NelderMead as nm

#Set to True if you want to optimize the parameters with NelderMead module. Statistics are then saved in the 'stats_optimized' directory
optimize = False

imDir = 'data_base/'
imExt = '.npy'
QDir = 'q_base/'
QExt = '_Q.npy'

#Getting all the images
images = sorted(os.listdir(imDir))
result = []

for img in images:
    #Loading dct of the image
    target = os.path.splitext(img)[0]
    imPath = os.path.join(imDir, target + imExt)
    dct = np.load(imPath)[0]

    #Loading quantization table of the dct   
    QPath = os.path.join(QDir, target + QExt)
    Q = np.load(QPath)[0]

    #Where we want to put the stats' graphs
    if optimize is True:
        saveDir = 'stats_optimized/'+ target
        saveDirRC = 'statsRC_optimized/' + target
    else:
        saveDir = 'stats/' + target
        saveDirRC = 'statsRC/' + target

    #Creating directory for the stats of the 'target' image
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        print("Directory ", saveDir, "Created. ")
    else:
        print("Directory ", saveDir, "already exists. ")

    if not os.path.exists(saveDirRC):
        os.makedirs(saveDirRC)
        print("Directory ", saveDirRC, "Created. ")
    else:
        print("Directory ", saveDirRC, "already exists. ")

    neg_output = np.ndarray((8,8), dtype=np.int8) #Check wether the parameters are valid or not
    neg_output[0][0] = True # dct mode (0,0) is not handled
    total_validity = 0

    for x in range(8):
        for y in range(8):
            if x != 0 or y != 0: #Excluding the DC coefficient
                print(x, y)
                #Creating the stat object
                statID = "{0}_({1}-{2})".format(target, x, y)
                statID = target+'('+str(x)+','+ str(y) +')'
                stat_test = stats.DCTstat(dct, Q, x, y)


                #getting and setting the optimized parameters
                if optimize is True:
                    optimizer = nm.ParamOptimizer(stat_test)
                    print(stat_test.eta, stat_test.nu)
                    """
                    if optimizer.opt.success:
                        stat_test.eta = optimizer.opt.x[0]
                        stat_test.nu = optimizer.opt.x[1]
                        print(stat_test.eta, stat_test.nu)
                    else:
                        print(optimizer.opt.message)
                    """
                #Computing the statistics with the optimized parameters
                stat_test.compute()


                #Checks the validity (greater or equal to zero) of each set of parameter. valid=True, invalid=False
                neg_output[x][y] = int((stat_test.eta >= 0) and (stat_test.nu >= 0))
                
                #Constructing the figure that shows the histogram, probability density function and probability mass function of the dataset
                plt.figure(num=statID)

                plt.hist(stat_test.data, bins=stat_test.L, density=True, label="eta = "+str(round(stat_test.eta, 3))+"  nu = "+str(round(stat_test.nu, 3))+"  Q = "+str(stat_test.Q))
                plt.plot(stat_test.L, stat_test.pmf, 'r+', label='PMF')
                plt.plot(stat_test.X, stat_test.Fi, label='PDF')
                plt.legend(loc='upper left')
                
                savePath = os.path.join(saveDir, statID+'.png')
                plt.savefig(savePath)
                
                plt.close()
    
    # Building a validity percentage of all the 64 sets of parameters for 1 image
    unique, counts = np.unique(neg_output, return_counts=True)
    dic = dict(zip(unique, counts))
    percent = dic[1] / 64
    total_validity += percent
    result.append( [ target + " : Valid Parameter Percentage:" + str(percent) ] )


result.append( [ "TOTAL valid Parameter Percentage: " + str( round( total_validity / len(images) ) ) ] )
print(*result, sep="\n")
