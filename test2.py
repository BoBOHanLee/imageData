# 灰階共生矩陣測試  12
import cv2
import math
import numpy as np

# 定義最大灰度級數
max_gray_level = 4

# 定義 P(i,j,d,theta)

def P(i,j,d,img,theta):
    height, width = img.shape
    sum = 0
    #零度
    if theta == 0 :
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                  if x < width-1 and img[y,x] == i and img[y,x+d] == j :   #跟右邊比
                      sum += 1
                  if  x > 0 and img[y,x] == i and img[y,x-d] == j :   #跟左邊比
                      sum += 1
        return sum


# read file
img_success = cv2.imread("Data_success/train_35.jpg", 0)
img_clean = cv2.imread("Data_fail/train_1420.jpg", 0)
img_fail = cv2.imread("Data_fail/train_21.jpg", 0)
img_test = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
img = img_fail



# gray level    0  -  255 變成 0   -   max-gray_level
#equ = cv2.equalizeHist(img)  # 先作值方圖均衡 讓灰階0至255都包含  #整張圖會跑掉故取消這一步驟
#print(img)
height,width = img.shape
scope = 256/max_gray_level
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        img[j,i] = (int)(img[j,i]/scope)

#計算灰階共生矩陣，這邊採用距離為1(d=1)，四種角度都計算
d = 1
# 0度
initial_glcm = np.zeros([max_gray_level, max_gray_level])
for  i in range(max_gray_level) :   #  i 從 0 到 max_gray_level
    for j in range(max_gray_level):
        #  #(i,j)
        initial_glcm[j,i] = P(i,j,d,img_test,0)






'''
cv2.imshow("look",img)
cv2.waitKey(0)
'''