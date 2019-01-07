import  cv2
import numpy as np
import pandas as pd


#HOG參數設定
winSize = (24,24)   #photo size
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
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType,
                                L2HysThreshold, gammaCorrection, nlevels, signedGradients)



database = pd.DataFrame()    #建立二維資料表的框架
dic_success = {'photo_numbers':2507 , 'file_name':"Data_success/train_{:.0f}.jpg" , 'label' : 0}   #photo_numbers 序號的最後一張(序號從0開始)
dic_fail = {'photo_numbers':2016 , 'file_name':"Data_fail/train_{:.0f}.jpg" , 'label' : 1}
dic_noExtusion = {'photo_numbers':147 , 'file_name':"Data_noExtusion/train_{:.0f}.jpg" , 'label' : 2}
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

        #建立hog特徵
        descriptor = hog.compute(img)
        s1 = np.reshape(descriptor, (1, 144))  #轉換形狀
        df_feature = pd.DataFrame(s1,  index=["{:.0f}".format(sum + n)])
        df_feature['label'] = label


        # 所有圖片資料彙整
        database = database.append(df_feature)  # combine in y-direction

    # 如此 index才會一直累積增加
    sum += photo_num
    sum += 1


#print(database)
#建立.csv檔
database.to_csv('database2.csv')