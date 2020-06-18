import jpeg_stats as stats
import NelderMead as nm
import numpy as np
import os
import matplotlib.pyplot as plt




optimize = True # Set to True to use NelderMead module


target = "00003.npy"
imDir = "data_base/"
dct = np.load(imDir+target)[0] # ton image
Q = np.load('q_base/00001_Q.npy')[0] # Chemin de ta 

#Le chemin où tu veux que les images soient créées
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

for x in range(8):
    for y in range(8):
        if x != 0 or y != 0: #Excluding the DC coefficient
            print(x, y)
            #Creating the stat object
            statID = "{0}_({1}-{2})".format(target, x, y)
            statID = target+'('+str(x)+','+ str(y) +')'
            stat_test = stats.DCTstat(dct, Q, x, y)
            stat_test.compute()


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
            


            #Checks the validity (greater or equal to zero) of each set of parameter. valid=True, invalid=False
            neg_output[x][y] = int((stat_test.eta >= 0) and (stat_test.nu >= 0))
            
            #Constructing the figure that shows the histogram, probability density function and probability mass function of the dataset
            """
            plt.figure(num=statID)

            plt.hist(stat_test.data,bins=stat_test.L, density=True, label="eta = "+str(round(stat_test.eta, 3))+"  nu = "+str(round(stat_test.nu, 3))+"  Q = "+str(stat_test.Q))
            plt.plot(stat_test.L, stat_test.pmf, 'r+', label='PMF')
            plt.plot(stat_test.X, stat_test.Fi, label='PDF')
            plt.legend(loc='upper left')
            
            savePath = os.path.join(saveDir, statID+'.png')
            plt.savefig(savePath)
            
            plt.close()
            """

            plt.figure(num=statID)

            PMF=stat_test.PvRC()
            plt.hist(stat_test.data, bins=stat_test.L  , density=True, label="eta = "+str(round(stat_test.eta, 3))+"  nu = "+str(round(stat_test.nu, 3))+"  Q = "+str(stat_test.Q))
            plt.plot(stat_test.X , stat_test.Fi, label='PDF')
            X = stat_test.L
            plt.plot(X, PMF, 'r+', label='PMF')
            PMFoptim = stat_test.PvRC(e=optimizer.opt.x[0] , n=optimizer.opt.x[1])
            plt.plot(X, PMFoptim, 'g+', label='PMFoptim')

            savePath = os.path.join(saveDirRC, statID+'.png')
            plt.savefig(savePath)



            plt.hist(stat_test.data, bins =np.arange( np.min(stat_test.data) , np.max(stat_test.data) )  , density=True, label="eta = "+str(round(stat_test.eta, 3))+"  nu = "+str(round(stat_test.nu, 3))+"  Q = "+str(stat_test.Q))
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
print( target + " : Valid Parameter Percentage:" + str(percent) )
