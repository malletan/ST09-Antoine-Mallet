import numpy as np
import jstegclass as jsteg

payload = 0.1


cover_img = np.load("data_base/images/100/00004.npy")[0] # cover image
print(cover_img.shape)
stego_img = jsteg.JSteg_Embedding(cover_img, payload) # stego image

qtable = np.load("data_base/qtables/100/00004_Q.npy")[0] # quant table
print(qtable.shape)


JTEST_cover = jsteg.IMGstat(cover_img, qtable, payload = payload)
JTEST_stego = jsteg.IMGstat(stego_img, qtable, payload = payload)

JTEST_cover.log_ratio_test()
JTEST_stego.log_ratio_test()

print("Normalized log-likelihood ratio test for the cover image: ", JTEST_cover.LRTnorm)
print("Expectation:{0}\t\tvariance:{1}".format(JTEST_cover.mu0, JTEST_cover.var0))
print("Normalized log-likelihood ratio test for the stego image: ", JTEST_stego.LRTnorm)
print("Expectation:{0}\t\tvariance:{1}".format(JTEST_stego.mu0, JTEST_stego.var0))

