# HOG features

import  cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


img = cv2.imread("fail.jpg",0)

'''
# Calculate gradient
im = np.float32(img) / 255.0
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
'''

winSize = np.shape(img)   #photo size
blockSize = (16, 16)     #用以解決照明問題   *需多方測試與比較 這邊暫用16 x 16
blockStride = (8, 8)     #Typically a blockStride is set to 50% of blockSize.
cellSize = (8, 8)       #計算HOG的基本mask  *需多方測試與比較 這邊暫用8 x 8
nbins = 9               #選轉角度的區間 0~180度  間隔20度
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True  #  0~360度變成 -180 ~ 180

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, signedGradients)

descriptor = hog.compute(img)
#print(np.shape(descriptor))  # 16*16 block run 24*24 pixeks  ---->  4 positions  ; eeach 16*16 block  contain 4 historigram ---> 36   => 4*36 = 144
#print(descriptor)

s1 = np.reshape(descriptor,(1,144))
df_feature = pd.DataFrame(s1)
print(df_feature)
df_feature['label'] = 0
print(df_feature)

'''
cv2.imshow("look",descriptor)
cv2.waitKey(0)
'''