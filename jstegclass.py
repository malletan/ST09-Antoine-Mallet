import numpy as np
import matplotlib.pyplot as plt
import jpeg_stats as stats


def JSteg_Embedding( dctCover , payload ):
    """Take a cover image and randomly embedds bits of 'hidden data' on it, thus creating a new 'stego' image."""
    dctTmp  = dctCover.flatten()                                                                    # En gros, pour indexer les arrays, je le met en vecteur
    dctChangeable = np.logical_and( dctTmp!=1 , dctTmp!=0 )                                         # On selectionne les coefs DCT different de 0 / 1
    r = np.random.permutation( np.sum( dctChangeable ) )                                            # On procede a une permutation des coefs pour changer l'order de lecture et etaler l'insertion
    nbChanges = int( np.min( [ np.round( payload/2 * len(dctTmp) ) , np.sum( dctChangeable ) ] ) )  # Calcul du nombre de changement, on s'assure que c'est pas plus grand que le nombre de coefs modifiables
    toChange = r[0:nbChanges] 
    changesToDo = (-1)**abs(dctTmp[ toChange ])                                                          # Ici gros trick, on utilise le fait que (-1)^x --> 1 si x est pair et -1 si x impair, c'est donc la modif ajoute pour simuler l'insertion
    dctTmp[ toChange ] = dctTmp[ toChange ] + changesToDo                                           # Insertion simulee ...
    dctStego = np.reshape( dctTmp , dctCover.shape)
    return(dctStego)


def getROC( GLR_H0 , GLR_H1 ):
    GLR_H0 = np.sort(GLR_H0)                                    # On va utiliser les donnees comme seuil: "adaptatif" pour calcule les probas de faux positif /  fausse-alarme (PFA) et la puissance (proba de vrais positifs)
    seuils = (GLR_H0[0:-1] + GLR_H0[1:]) /2                     # On prend les seuils entre deux valeurs successives de test pour les images cover
    PFA = np.arange(len(seuils)+1, -1, -1) / ( len(seuils)+1 )  # Avec ces valeurs de seuils, le PFA est simplemenet 0 1/N 2/N ..... 1  ; les seuils sont croissants donc le PFA en fonction du seuil est decroissante ....
    Power = np.zeros(len(seuils)+2)                             # pre-allocation 
    Power[0] = 1
    Idx=1
    nbH1 = len( GLR_H1 )
    for seuil in seuils:
        #print( seuil, Idx,  np.sum( GLR_H1 > seuil)  / nbH1 , PFA[Idx] )
        try :
            Power[Idx] = np.sum( GLR_H1 > seuil) / nbH1             # Pour chaque valeur de seuil, on calcul combien de valeur de LR/GLR pour les stego sont au dessus du seuil (donc classee correctoment comme stego)
        except RuntimeWarning:
            print(seuil)
        Idx = Idx + 1
    plt.plot( PFA, Power)
    plt.axis([0, 1, 0.0, 1])	                                #J'ai ajuste les axes pour mieux "voir" ce qui se passe
    plt.xlabel(r'$\alpha_0$ --- false positive rate', fontsize=35)
    plt.ylabel(r'$\beta_1$ --- detection rate', fontsize=35)
    plt.show()
    return( PFA , Power )



class IMGstat:
    """The purpose of this class is to gather all the dct-statistics from 1 image (from the 63 AC modes),
       and then building the required statistics to conduct the delta star test to check wether the image is detected as
       a cover image or a stego one."""

    def __init__(self, dct, qmtx, x = range(8), y = range(8), payload = None):
        """The input array is expected to be a 2-D array of the DCTed image we want to test."""
        self.vk = [] #contains the vectorized dct coefficients
        self.vkstat = [] # contains the DCTstat objects
        self.x = x # Number of lines (for each dct block) to take into account for the test
        self.y = y # Number of rows (for each dct block) to take into account for the test
        self.mu0 = 0
        self.var0 = 0
        for x in self.x:
            for y in self.y:
                if x != 0 or y != 0:
                    q = qmtx[x][y]
                    vki = dct[x*8+y,:]
                    if np.sum( np.array(vki)!=0 ) > 10:
                        vki_stat = stats.DCTstat(vki, q, x, y, payload)
#                        print( vki_stat.eta , vki_stat.nu )

                        vki_stat.optimize_parameters()
#                        print( vki_stat.eta , vki_stat.nu )
#                        vki_stat.compute()
    
                        self.vkstat.append( vki_stat )
                        self.vk.append( vki )


    def log_ratio_test(self):
        """We start by computing the LAMBDA(V) as expressed in Eq. (47)."""
        LRT = 0
        validity = 0
        for i in range(len(self.vkstat)):
            if self.vkstat[i].eta > 0 and self.vkstat[i].nu > 0:
                self.mu0 += self.vkstat[i].esp0()
                self.var0 += self.vkstat[i].var0()
            
                LRT += self.vkstat[i].logRatio()
                #print(i ,'/', len(self.vkstat) , ' \t' , LRT, self.mu0, self.var0 )
                validity += 1
        
        self.LRT = LRT
        self.validity = validity
        #We then computes the LAMBDA*(V) as expressed in Eq. (65).
        self.LRTnorm = ( LRT - self.mu0 ) / np.sqrt(self.var0)


