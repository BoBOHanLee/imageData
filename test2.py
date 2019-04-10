# 灰階共生矩陣測試  12
import cv2
import math
import numpy as np
from sklearn import preprocessing

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
                  if x+d < width and img[y,x] == i and img[y,x+d] == j :   #跟右邊比
                      sum += 1
                  if  x-d > -1 and img[y,x] == i and img[y,x-d] == j :   #跟左邊比
                      sum += 1
        return sum


    #45度
    if theta == 45:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x-d > -1 and y+d < height  ) and img[y, x] == i and img[y + d, x - d] == j:  # 跟左下比
                    sum += 1
                if (y-d > -1 and x+d < width  ) and img[y, x] == i and img[y - d, x + d] == j:  # 跟右上比
                    sum += 1
        return sum



     # 90度
    if theta == 90:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if y+d < height  and img[y , x] == i and img[y + d, x ] == j:  # 跟下面比
                    sum += 1
                if y-d > -1 and img[y, x] == i and img[y - d, x ] == j:  # 跟上面比
                    sum += 1
        return sum

    # 135度
    if theta == 135:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x+d < width  and y+d < height ) and img[y, x] == i and img[y + d, x + d] == j:  # 跟右下比
                    sum += 1
                if (y-d > -1 and x-d > -1) and img[y, x] == i and img[y - d, x - d] == j:  #  跟左上比
                    sum += 1
        return sum



def normalize_glcm(img,initial,theta):   #指定d = 1
    height, width = img.shape

    if theta == 0:
        total =  2*height*(width - 1) #水平排列總次數

    if theta == 45:
        total =  2*(height - 1)*(width - 1) #右上左下排列總次數

    if theta == 90:
        total =  2*width*(height-1) #垂直排列總次數

    if theta == 135:
        total =  2*(height - 1)*(width - 1) #左上右下排列總次數


    return initial/total




# read file
img_success_black = cv2.imread("Data_success/train_3356.jpg", 0)
img_success_white = cv2.imread("Data_success/train_10429.jpg", 0)

img_fail_black = cv2.imread("Data_fail/train_12.jpg", 0)
img_fail_white = cv2.imread("Data_fail/train_12631.jpg", 0)

img_clean = cv2.imread("Data_noExtusion/train_4719.jpg", 0)

img_test = np.array([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
img = img_success_white



# gray level    0  -  255 變成 0   -   max-gray_level
#equ = cv2.equalizeHist(img)  # 先作值方圖均衡 讓灰階0至255都包含  #整張圖會跑掉故取消這一步驟
#print(img)
height,width = img.shape
scope = 256/max_gray_level
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        img[j,i] = (int)(img[j,i]/scope)

#計算灰階共生矩陣，這邊採用距離為1(d=1)，四種角度都計算
d = 2
theta = 90
# 建立灰階共生矩陣
initial_glcm = np.zeros([max_gray_level, max_gray_level])
for  i in range(max_gray_level) :   #  i 從 0 到 max_gray_level-1
    for j in range(max_gray_level):
        #  #(i,j)
        #initial_glcm[j,i] = P(i,j,d,img_test,135)
        initial_glcm[j, i] = P(i, j, d,img_test, theta)


print(initial_glcm)
#將灰階共生矩陣規範化    把count(計數)轉變為probability(機率)
print('normalize')
#glcm = preprocessing.normalize(initial_glcm,norm = 'l2')  #錯誤
glcm = normalize_glcm(img,initial_glcm,theta)
print(glcm)


#  ASM （angular second moment)特征（或称能量特征）
height,width = glcm.shape
sum_asm = 0.0
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        a =  glcm[j,i]* glcm[j,i]
        sum_asm += a

print("asm = {:f}".format(sum_asm))



# 对比度（Contrast）
sum_contrast = 0.0
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        a =  (i - j)*(i - j)*glcm[j,i]
        sum_contrast += a

print("contrast = {:f}".format(sum_contrast))


# 熵（entropy）
sum_entropy = 0.0
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        if glcm[j,i] != 0 :
          a =  -1*glcm[j,i]*np.log(glcm[j,i])
          sum_entropy += a

print("entropy = {:f}".format(sum_entropy))

'''
# 自相关（correlation）   算法不確定 先展緩
sum_correlation = 0
px=[0,0,0,0]
py=[0,0,0,0]
for j in range(height):
    px  += glcm[:, j]
for i in range(width):
    py  += glcm[i, :]
'''

#  逆差矩（IDM：Inverse Difference Moment）
sum_idm = 0
for  i in range(height) :   #  i 從 0 到 height-1
    for j in range(width):
        sum_idm += (1/(1+(i-j)*(i-j)))*glcm[j,i]

print("idm = {:f}".format(sum_idm))



'''
cv2.imshow("look",img)
cv2.waitKey(0)
'''