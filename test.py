import  cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool


#值方圖找特徵


# read
img_success = cv2.imread("Data_success/train_35.jpg",0)
img_clean = cv2.imread("Data_fail/train_1420.jpg",0)
img_fail  = cv2.imread("Data_fail/train_21.jpg",0)
img_noEx = cv2.imread("Data_noExtusion/train_5.jpg",0)

img_clean2 = cv2.imread("Data_fail/train_12.jpg",0)
img_clean3 = cv2.imread("Data_fail/train_31.jpg",0)
img_clean4 = cv2.imread("Data_fail/train_49.jpg",0)
img_clean5 = cv2.imread("Data_fail/train_75.jpg",0)
img_clean6 = cv2.imread("Data_fail/train_85.jpg",0)
img_clean7 = cv2.imread("Data_fail/train_111.jpg",0)
img_clean8 = cv2.imread("Data_fail/train_151.jpg",0)
img_clean9 = cv2.imread("Data_fail/train_146.jpg",0)
img_clean10 = cv2.imread("Data_fail/train_153.jpg",0)
img_clean11= cv2.imread("Data_fail/train_174.jpg",0)

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





'''
# the most num
print(calMostNum(img_clean))
print(calMostNum(img_clean2))
print(calMostNum(img_clean3))
print(calMostNum(img_clean4))
print(calMostNum(img_clean5))
print(calMostNum(img_clean6))
print(calMostNum(img_clean7))
print(calMostNum(img_clean8))
print(calMostNum(img_clean9))
print(calMostNum(img_clean10))
print(calMostNum(img_clean11))
'''



'''
#SD

print(calSD(img_clean))
print(calSD(img_clean2))
print(calSD(img_clean3))
print(calSD(img_clean4))
print(calSD(img_clean5))
print(calSD(img_clean6))
print(calSD(img_clean7))
print(calSD(img_clean8))
print(calSD(img_clean9))
print(calSD(img_clean10))
print(calSD(img_clean11))
print(calSD(img_success))
print(calSD(img_fail))
'''






'''
# medium
print(calMidium(img_clean))
print(calMidium(img_clean2))
print(calMidium(img_clean3))
print(calMidium(img_clean4))
print(calMidium(img_clean5))
print(calMidium(img_clean6))
print(calMidium(img_clean7))
print(calMidium(img_clean8))
print(calMidium(img_clean9))
print(calMidium(img_clean10))
print(calMidium(img_clean11))
'''




'''
# average
print(calAverage(img_clean))
print(calAverage(img_clean2))
print(calAverage(img_clean3))
print(calAverage(img_clean4))
print(calAverage(img_clean5))
print(calAverage(img_clean6))
print(calAverage(img_clean7))
print(calAverage(img_clean8))
print(calAverage(img_clean9))
print(calAverage(img_clean10))
print(calAverage(img_clean11))
'''

'''

# entropy
print(calcEntropy(img_success))
print(calcEntropy(img_clean))
print(calcEntropy(img_fail))

print(calcEntropy(img_clean2))
print(calcEntropy(img_clean3))
print(calcEntropy(img_clean4))
print(calcEntropy(img_clean5))
print(calcEntropy(img_clean6))
print(calcEntropy(img_clean7))
print(calcEntropy(img_clean8))
print(calcEntropy(img_clean9))
print(calcEntropy(img_clean10))
print(calcEntropy(img_clean11))
'''

'''
# histogram
#print(calSD(img_success))
#print(calSD(img_clean))
#print(calSD(img_fail))

hist_suc = np.bincount(img_success.ravel(), minlength=256)
hist_clean = np.bincount(img_clean.ravel(), minlength=256)
hist_fail = np.bincount(img_fail.ravel(), minlength=256)
plt.plot(hist_suc)
plt.plot(hist_clean)
plt.plot(hist_fail)
plt.show()
'''


'''
# histogram
hist_clean = np.bincount(img_clean.ravel(), minlength=256)
hist_clean2 = np.bincount(img_clean2.ravel(), minlength=256)
hist_clean3 = np.bincount(img_clean3.ravel(), minlength=256)
hist_clean4 = np.bincount(img_clean4.ravel(), minlength=256)
hist_clean5 = np.bincount(img_clean5.ravel(), minlength=256)
hist_clean6 = np.bincount(img_clean6.ravel(), minlength=256)
hist_clean7 = np.bincount(img_clean7.ravel(), minlength=256)
hist_clean8 = np.bincount(img_clean8.ravel(), minlength=256)
hist_clean9 = np.bincount(img_clean9.ravel(), minlength=256)
hist_clean10 = np.bincount(img_clean10.ravel(), minlength=256)
hist_clean11 = np.bincount(img_clean11.ravel(), minlength=256)
plt.plot(hist_clean)
plt.plot(hist_clean2)
plt.plot(hist_clean3)
plt.plot(hist_clean4)
plt.plot(hist_clean5)
plt.plot(hist_clean6)
plt.plot(hist_clean7)
plt.plot(hist_clean8)
plt.plot(hist_clean9)
plt.plot(hist_clean10)
plt.plot(hist_clean11)
plt.show()
'''


'''
# histogram
hist_success = np.bincount(img_success.ravel(), minlength=256)
hist_clean = np.bincount(img_clean.ravel(), minlength=256)
hist_fail = np.bincount(img_fail.ravel(), minlength=256)
hist_noEx = np.bincount(img_noEx.ravel(), minlength=256)
plt.plot(hist_success,label='success')
plt.plot(hist_clean,label='background')
plt.plot(hist_fail,label='fail')
plt.plot(hist_noEx,label='noEx')
plt.legend(loc='upper right')
plt.show()
'''


for n in range(2508) :
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