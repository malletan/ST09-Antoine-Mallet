import numpy as np
import matplotlib.pyplot as plt
import jpeg_stats as stats


def JSteg_Embedding( dctCover , payload ):
    dctTmp  = dctCover.flatten()                                                                    # En gros, pour indexer les arrays, je le met en vecteur
    dctChangeable = np.logical_and( dctTmp!=1 , dctTmp!=0 )                                         # On selectionne les coefs DCT different de 0 / 1
    r = np.random.permutation( np.sum( dctChangeable ) )                                            # On procede a une permutation des coefs pour changer l'order de lecture et etaler l'insertion
    nbChanges = int( np.min( [ np.round( payload/2 * len(dctTmp) ) , np.sum( dctChangeable ) ] ) )  # Calcul du nombre de changement, on s'assure que c'est pas plus grand que le nombre de coefs modifiables
    toChange = r[0:nbChanges] 
    changesToDo = (-1)**dctTmp[ toChange ]                                                          # Ici gros trick, on utilise le fait que (-1)^x --> 1 si x est pair et -1 si x impair, c'est donc la modif ajoute pour simuler l'insertion
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
        Power[Idx] = np.sum( GLR_H1 > seuil) / nbH1             # Pour chaque valeur de seuil, on calcul combien de valeur de LR/GLR pour les stego sont au dessus du seuil (donc classee correctoment comme stego)
        Idx = Idx + 1
    plt.plot( PFA, Power)
    plt.axis([0, 1, 0.0, 1])	                                #J'ai ajuste les axes pour mieux "voir" ce qui se passe
    plt.xlabel(r'$\alpha_0$ --- false positive rate', fontsize=35)
    plt.ylabel(r'$\beta_1$ --- detection rate', fontsize=35)
    #plt.show()
    return( PFA , Power )


payload = 0.1
dctCover =  np.round( np.random.normal( 1 , 4 , [256,256] ) ) 
dctStego = JSteg_Embedding( dctCover , payload )
print( np.sum( dctCover != dctStego) )

score0 = np.random.normal(0,1,1000)
score1 = np.random.normal(1.5,1,1000)

PFA, PWR = getROC( score0 , score1 )

R = payload



"""
Q = np.ones([8,8], dtype=np.int32)
S = stats.DCTstat(dctCover.astype(np.int32), Q, 1, 1)
S.compute()
Q_r = S.Qr(R)
print(S.pmf)
print(Q_r)

plt.figure(2)
plt.plot(S.L, S.pmf, '+r')
plt.plot(S.L, Q_r, 'xb')
plt.show()

#Q_r_theta = (1 - R) * S.Pv(S.L) + R / 2 * S.Pv()
"""