
from keras.utils import to_categorical

import pickle
import numpy as np

from DataPreproc import Data_pre


def index_label(label, labels_val):
    # print(label)
    list = []
    for j in range(len(label)):
        for i in range(len(labels_val)):
            if label[j] == labels_val[i]:
                list.append(i)
                break
    return np.asarray(list)


# Img_Path = "C:\\Users\\aiia\\Desktop\\hanja_train\\"
Img_Path = "C:\\Users\\aiia\\Desktop\\210701_hanja_zip\\ITKC_MO_0628A_A310\\"

img_h, img_w, chanel = 100, 100, 1
Dp = Data_pre(img_h, img_w, chanel)
x_train, y_train, labels_val, label_dic, name  = Dp.data_set_fun(Img_Path, 0)

print(y_train)
print(name)

with open('E:\\test_hanja_label.pkl', 'wb') as file_pi:
    pickle.dump(name, file_pi, protocol=4)

# y_train = to_categorical(y_train)
# print(y_train)
# class_num = len(labels_val)
# print(np.shape(x_train))
# print(np.shape(y_train))
# print(labels_val)
# # print(class_num)
# print(label_dic)

with open('E:\\test_hanja.pkl', 'wb') as file_pi:
    pickle.dump(x_train, file_pi, protocol=4)



############ load test ####################


# with open( "E:\\train_data.pkl", "rb" ) as file:
#     loaded_data = pickle.load(file)
#     print(np.shape(loaded_data))
#
# with open( "E:\\train_data2.pkl", "rb" ) as file:
#     loaded_data2 = pickle.load(file)
#     print(np.shape(loaded_data2))
#
#
# sum_data = np.concatenate((loaded_data,loaded_data2), axis=0)
# print(sum_data)
# print(np.shape(sum_data))


# with open( "E:\\train_data_label.pkl", "rb" ) as file:
#     loaded_data = pickle.load(file)
#     print(np.shape(loaded_data))
#     print(loaded_data)
#
# with open( "E:\\train_data_label2.pkl", "rb" ) as file:
#     loaded_data2 = pickle.load(file)
#     print(np.shape(loaded_data2))
#     print(loaded_data2)
#
# sum_data_label = np.concatenate((loaded_data,loaded_data2), axis=0)
# print(sum_data_label)
# print(np.shape(sum_data_label))
#
# labels_val = list(set(sum_data_label))
# labels_val.sort()
# print(labels_val)
# Y_set = index_label(sum_data_label, labels_val)
# print(Y_set)
#
# y_train = to_categorical(Y_set)
# print(y_train)