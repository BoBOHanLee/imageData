# 建立以值方圖以及灰階共生矩陣為主的database

import  cv2
import numpy as np
import pandas as  pd


#-------------------值方圖特徵計算方程式-----------------------------#

def calAverage(hist,total_pixel):
    sum = 0
    i = 0   #gray level
    for item in hist:
        sum = sum + item*i
        i += 1

    return sum/total_pixel


def calEntropy(hist,total_pixel):
    entropy = []
    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en


def calMostNum(hist):
    i = 0  # gray level
    cup = 0
    cup_i=0
    for item in hist:
        if item[0]>=cup :
          cup = item[0]
          cup_i = i

        i += 1
    return cup_i


def calSD(hist,average):
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


    return  np.sqrt(sum/(255+1))[0]   #[0]為了從numpy的矩陣中取直


def calVariation(average,std):
    num = std/average
    return num[0]  #[0]為了從numpy的矩陣中取直

#--------------------------------------------------------------------#



#----------------灰階共生矩陣特徵計算方程式--------------------------#
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


    #45度
    if theta == 45:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x > 0 and y < height - 1 ) and img[y, x] == i and img[y + d, x - d] == j:  # 跟左下比
                    sum += 1
                if (y > 0 and x < width - 1 ) and img[y, x] == i and img[y - d, x + d] == j:  # 跟右上比
                    sum += 1
        return sum



     # 90度
    if theta == 90:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if y < height - 1 and img[y , x] == i and img[y + d, x ] == j:  # 跟下面比
                    sum += 1
                if y > 0 and img[y, x] == i and img[y - d, x ] == j:  # 跟上面比
                    sum += 1
        return sum

    # 135度
    if theta == 135:
        for y in range(height):  # y 從 0 到 height-1
            for x in range(width):
                if (x < width - 1 and y < height - 1) and img[y, x] == i and img[y + d, x + d] == j:  # 跟右下比
                    sum += 1
                if (y > 0 and x > 0) and img[y, x] == i and img[y - d, x - d] == j:  #  跟左上比
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



def create_GLCM(img):
    #灰階0~255先分階層
    max_gray_level = 4  # 定義最大灰度級數
    height, width = img.shape
    scope = 256 / max_gray_level
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            img[j, i] = (int)(img[j, i] / scope)

    # 計算灰階共生矩陣，這邊採用距離為1(d=1)，0角度
    d = 1
    theta = 0
    # 建立灰階共生矩陣
    initial_glcm = np.zeros([max_gray_level, max_gray_level])
    for i in range(max_gray_level):  # i 從 0 到 max_gray_level-1
        for j in range(max_gray_level):
            #  #(i,j)
            # initial_glcm[j,i] = P(i,j,d,img_test,135)
            initial_glcm[j, i] = P(i, j, d, img, theta)

    # 將灰階共生矩陣規範化    把count(計數)轉變為probability(機率)
    glcm = normalize_glcm(img, initial_glcm, theta)

    return  glcm


# ASM （angular second moment)特征（或称能量特征）
def cal_asm(glcm,height,width):
    sum_asm = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            a = glcm[j, i] * glcm[j, i]
            sum_asm += a

    #print("asm = {:f}".format(sum_asm))
    return sum_asm


# 对比度（Contrast）
def cal_contrast(glcm,height,width):
    sum_contrast = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            a = (i - j) * (i - j) * glcm[j, i]
            sum_contrast += a

    #print("contrast = {:f}".format(sum_contrast))
    return sum_contrast

# 熵（entropy）
def cal_GLCMentropy(glcm,height,width):
    sum_entropy = 0.0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            if glcm[j, i] != 0:
                a = -1 * glcm[j, i] * np.log(glcm[j, i])
                sum_entropy += a

    #print("entropy = {:f}".format(sum_entropy))
    return sum_entropy

#  逆差矩（IDM：Inverse Difference Moment）
def cal_IDM(glcm,height,width):
    sum_idm = 0
    for i in range(height):  # i 從 0 到 height-1
        for j in range(width):
            sum_idm += (1 / (1 + (i - j) * (i - j))) * glcm[j, i]

    #print("idm = {:f}".format(sum_idm))
    return  sum_idm


#--------------------------------------------------------------------#

'''
database = pd.DataFrame()    #建立二維資料表的框架
for n in range(2508) :   #0~2507
    # read
    filename = "Data_success/train_{:.0f}.jpg".format(n)
    img = cv2.imread(filename,0)
    print(n)

    #直方圖特徵計算
    # 四變量計算
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]
    average = calAverage(hist, total_pixel)

    entropy = calEntropy(hist,total_pixel)
    mostNum = calMostNum(hist)
    std = calSD(hist, average)
    val = calVariation(average,std)
    HGfeatures = [[mostNum,std,val,entropy]] # 1 x 4
    #創表格，橫向特徵
    df_HGfeatures = pd.DataFrame(HGfeatures, columns=['HG_mostNum', 'HG_std','HG_val','HG_entropy'],index = ["{:.0f}".format(n)])


    #灰階共生矩陣特徵計算
    glcm = create_GLCM(img)
    height, width = glcm.shape

    asm = cal_asm(glcm,height,width)
    contrast = cal_contrast(glcm,height,width)
    GLCMentropy = cal_GLCMentropy(glcm,height,width)
    idm = cal_IDM(glcm,height,width)
    GLCMfeatures = [[asm,contrast,GLCMentropy,idm,0]]  #label 給定 success = 0
    # 創表格，橫向特徵   並且給定label(success = 0 ; fail = 1 ;noExtuition = 2 )
    df_GLCMfeatures = pd.DataFrame(GLCMfeatures, columns=['GLCM_asm', 'GLCM_contrast', 'GLCM_entropt', 'GLCM_idm' , 'label'],
                                 index=["{:.0f}".format(n)])


    #所有特徵彙整
    df_all = pd.concat([df_HGfeatures, df_GLCMfeatures], 1)  # combine in x-direction


    #所有圖片資料彙整
    database = database.append(df_all)  # combine in y-direction

print(database)
'''

database = pd.DataFrame()    #建立二維資料表的框架
dic_success = {'photo_numbers':17081 , 'file_name':"Data_success/train_{:.0f}.jpg" , 'label' : 0}   #photo_numbers 序號的最後一張(序號從0開始)
dic_fail = {'photo_numbers':17194 , 'file_name':"Data_fail/train_{:.0f}.jpg" , 'label' : 1}
dic_noExtusion = {'photo_numbers':4856 , 'file_name':"Data_noExtusion/train_{:.0f}.jpg" , 'label' : 2}
dic = [dic_success,dic_fail,dic_noExtusion]
# for index
sum = 0

for i in dic :
    photo_num = i['photo_numbers']
    filename = i['file_name']
    label = i['label']


    for n in range(photo_num+1):
        # read
        Filename = filename.format(n)
        img = cv2.imread(Filename, 0)
        print(n)

        # 直方圖特徵計算
        # 四變量計算
        hist = cv2.calcHist([img], [0], None, [256], [0, 255])
        total_pixel = img.shape[0] * img.shape[1]
        average = calAverage(hist, total_pixel)

        entropy = calEntropy(hist, total_pixel)
        mostNum = calMostNum(hist)
        std = calSD(hist, average)
        val = calVariation(average, std)
        HGfeatures = [[mostNum, std, val, entropy]]  # 1 x 4
        # 創表格，橫向特徵
        df_HGfeatures = pd.DataFrame(HGfeatures, columns=['HG_mostNum', 'HG_std', 'HG_val', 'HG_entropy'],
                                     index=["{:.0f}".format(sum + n )])

        # 灰階共生矩陣特徵計算
        glcm = create_GLCM(img)
        height, width = glcm.shape

        asm = cal_asm(glcm, height, width)
        contrast = cal_contrast(glcm, height, width)
        GLCMentropy = cal_GLCMentropy(glcm, height, width)
        idm = cal_IDM(glcm, height, width)
        GLCMfeatures = [[asm, contrast, GLCMentropy, idm, label]]  # label 給定 success = 0
        # 創表格，橫向特徵   並且給定label(success = 0 ; fail = 1 ;noExtuition = 2 )
        df_GLCMfeatures = pd.DataFrame(GLCMfeatures,
                                       columns=['GLCM_asm', 'GLCM_contrast', 'GLCM_entropt', 'GLCM_idm', 'label'],
                                       index=["{:.0f}".format(sum + n )])

        # 所有特徵彙整
        df_all = pd.concat([df_HGfeatures, df_GLCMfeatures], 1)  # combine in x-direction

        # 所有圖片資料彙整
        database = database.append(df_all)  # combine in y-direction

    # 如此 index才會一直累積增加
    sum += photo_num
    sum += 1


#print(database)

#建立.csv檔
database.to_csv('database.csv')