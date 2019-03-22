import  cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool


#值方圖找特徵


# read
img_success_black = cv2.imread("Data_success/train_7395.jpg",0)
img_clean = cv2.imread("Data_noExtusion/train_32.jpg",0)
img_fail_black  = cv2.imread("Data_fail/train_211.jpg",0)

img_success_white= cv2.imread("Data_success/train_12150.jpg",0)
img_clean = cv2.imread("Data_noExtusion/train_32.jpg",0)
img_fail_white  = cv2.imread("Data_fail/train_12502.jpg",0)


'''
#threshold
__, img_clean = cv2.threshold(img_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
__, img_fail = cv2.threshold(img_fail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
'''

def calEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


def calAverage(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]
    sum = 0
    i = 0   #gray level
    for item in hist:
        sum = sum + item*i
        i += 1


    return sum/total_pixel


def calMidium(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]
    sum = 0
    i = 0  # gray level

    for item in hist:
       sum = sum + item
       if sum >= (total_pixel/2) :
           return i
       else:
           i += 1



def calSD(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    average = calAverage(img)
    sum = 0
    i = 0  # gray level

    for item in hist:
        if item[0] == 0:
            i += 1
            continue

        else:
             for x in range(1,int(item[0])) :
                 sum = sum + (i-average)**2
             i += 1

    return  np.sqrt(sum/(255+1))


def calMostNum(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    i = 0  # gray level
    cup = 0
    cup_i=0
    for item in hist:
        if item[0]>=cup :
          cup = item[0]
          cup_i = i

        i += 1


    return cup_i

def calVariation(img):
    average = calAverage(img)
    std =calSD(img)

    return std/average


'''
#variatiom
print(calVariation(img_clean))
print(calVariation(img_clean2))
print(calVariation(img_clean3))
print(calVariation(img_clean4))
print(calVariation(img_clean5))
print(calVariation(img_clean6))
print(calVariation(img_clean7))
print(calVariation(img_clean8))
print(calVariation(img_clean9))
print(calVariation(img_clean10))
print(calVariation(img_clean11))
print(calVariation(img_success))
print(calVariation(img_fail))
'''






# histogram
#print(calSD(img_success))
#print(calSD(img_clean))
#print(calSD(img_fail))
'''
hist_suc = np.bincount(img_success.ravel(), minlength=256)
hist_clean = np.bincount(img_clean.ravel(), minlength=256)
hist_fail = np.bincount(img_fail.ravel(), minlength=256)
#plt.plot(hist_suc)
#plt.plot(hist_clean)
plt.plot(hist_fail)
plt.show()
'''


'''
# histogram
hist_success = np.bincount(img_success_white.ravel(), minlength=256)
hist_clean = np.bincount(img_clean.ravel(), minlength=256)
hist_fail = np.bincount(img_fail_white.ravel(), minlength=256)
plt.plot(hist_success,label='success')
plt.plot(hist_clean,label='background')
plt.plot(hist_fail,label='fail')
plt.legend(loc='upper right')
plt.show()
'''


for n in range(17081) :
    sum = 0
    filename = "Data_success/train_{:.0f}.jpg".format(n)
    img = cv2.imread(filename,0)
    #四變量比較
    #entropy
    entropy = calEntropy(img)
    if entropy>=4 and entropy<=5:
        sum =sum +1
    #most num
    mostNum = calMostNum(img)
    if mostNum>=100 and mostNum<=150:
        sum += 1
    #variation
    val = calVariation(img)
    if val<=0.1:
        sum += 1
    #std
    std = calSD(img)
    if std>=5 and std<=10 :
        sum += 1

    if sum==4:
        print(filename)




#cv2.imshow("test",np.hstack([img_clean,img_fail]))
#cv2.waitKey(0)