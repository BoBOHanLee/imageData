import  cv2
import numpy as np
import pandas as  pd

database = pd.DataFrame()    #建立二維資料表的框架
dic_success = {'photo_numbers':2474 , 'file_name':"Data_success/train_{:.0f}.jpg" , 'label' : 0}   #photo_numbers 序號的最後一張(序號從0開始)
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

        #建立pixel矩陣
        s1 = np.reshape(img, (1, 576))  #轉換形狀  24*24 = 576
        df_feature = pd.DataFrame(s1, index=["{:.0f}".format(sum + n)])
        df_feature['label'] = label


        # 所有圖片資料彙整
        database = database.append(df_feature)  # combine in y-direction

    # 如此 index才會一直累積增加
    sum += photo_num
    sum += 1


#print(database)
#建立.csv檔
database.to_csv('database3.csv')