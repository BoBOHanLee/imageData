import  cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pool


#值方圖找特徵

# read
'''
img_success_black = cv2.imread("Data_success/train_7395.jpg",0)
img_clean = cv2.imread("Data_noExtusion/train_32.jpg",0)
img_fail_black  = cv2.imread("Data_fail/train_211.jpg",0)

img_success_white= cv2.imread("Data_success/train_12150.jpg",0)
img_clean = cv2.imread("Data_noExtusion/train_32.jpg",0)
img_fail_white  = cv2.imread("Data_fail/train_12502.jpg",0)

black_success1 = cv2.imread("Data_fail/train_63.jpg",0)
black_success2 = cv2.imread("Data_fail/train_64.jpg",0)
black_success3 = cv2.imread("Data_fail/train_66.jpg",0)
black_success4 = cv2.imread("Data_fail/train_1746.jpg",0)
black_success5 = cv2.imread("Data_fail/train_1748.jpg",0)
black_success6 = cv2.imread("Data_fail/train_1749.jpg",0)

hist = cv2.calcHist([img_clean], [0], None, [256], [0, 255])
print(hist/256)
'''
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
hist_suc1 = np.bincount(black_success1.ravel(), minlength=256)
hist_suc2 = np.bincount(black_success2.ravel(), minlength=256)
hist_suc3 = np.bincount(black_success3.ravel(), minlength=256)
hist_suc4 = np.bincount(black_success4.ravel(), minlength=256)
hist_suc5 = np.bincount(black_success5.ravel(), minlength=256)
hist_suc6 = np.bincount(black_success6.ravel(), minlength=256)

plt.plot(hist_suc1)
plt.plot(hist_suc2)
plt.plot(hist_suc3)
plt.plot(hist_suc4)
plt.plot(hist_suc5)
plt.plot(hist_suc6)
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


num = 0
for n in range(5602) :
    sum = 0
    filename = "collected_data/Black_success/train_{:.0f}.jpg".format(n)
    img = cv2.imread(filename,0)

    try:
        img.shape
    except:
        #print('fail to read  {:n}'.format(n))
        continue


    #改黨名
    filename_new = "collected_data/Black_fail/train_{:.0f}.jpg".format(num)
    #filename_new = "Data_success/train_{:.0f}.jpg".format(num)
    os.rename(filename,filename_new)
    num += 1
    print(num)



    #四變量比較

    entropy = calEntropy(img)
    mostNum = calMostNum(img)
    val = calVariation(img)
    std = calSD(img)
    '''
    #對背景而言
    if entropy>=0.1 and entropy<=5:
        sum =sum +1
    if mostNum>=0 and mostNum<=120:
        sum += 1
    if val<=0.1:
        sum += 1
    if std>=0 and std<=15 :
       sum += 1
    '''
    '''
    # 對完全黑而言
    if entropy >= 1 and entropy <= 5.2:
        sum = sum + 1
    if mostNum >= 0 and mostNum <=50:
        sum += 1
    if val <= 1.4:
        sum += 1
    if std >= 2 and std <= 20:
        sum += 1
    '''
    
    if sum==4:
        #刪除檔案
        os.remove(filename)


        print(filename)




#cv2.imshow("test",np.hstack([img_clean,img_fail]))
#cv2.waitKey(0)