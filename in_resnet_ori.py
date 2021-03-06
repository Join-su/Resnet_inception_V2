from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from keras.layers import Lambda, Input, Dense, Activation, ZeroPadding2D
from keras.models import Model, load_model, Sequential
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras.optimizers as optimizers
from keras import applications
from keras.utils import multi_gpu_model
from keras.utils import to_categorical

from PIL import Image
import cv2
import argparse
import os
import pickle
import numpy as np
import matplotlib.pylab as plt
from math import trunc
import shutil

from DataPreproc import Data_pre

# # ImageNet classification
# model = InceptionResNetV2()
# model.predict(...)

# # Finetuning on another 100-class dataset
# base_model = InceptionResNetV2(include_top=False, pooling='avg')
# outputs = Dense(100, activation='softmax')(base_model.output)
# model = Model(base_model.inputs, outputs)
# model.compile(...)
# model.fit(...)

train_run = 0 # 1: train, 0: test

svae_path = "C:\\Users\\aiia\\Desktop\\210701_hanja_zip\\ITKC_MO_0066A_A012\\"

img_h, img_w, chanel = 100, 100, 1
img_shape = (img_h,img_w,chanel)
epochs = 20



def index_label(label, labels_val):
    # print(label)
    list = []
    for j in range(len(label)):
        for i in range(len(labels_val)):
            if label[j] == labels_val[i]:
                list.append(i)
                break
    return np.asarray(list)


def record_func(epoch, logs):
    if epoch % 10 == 0:
        model.save(svae_path + '%d_AE_model.h5' % epoch)

    i = logs.get('acc')
    f = open("record_acc.txt", 'a')
    data = "{}\n".format(i)
    f.write(str(data))
    f.close()

    i = logs.get('loss')
    f = open("record_loss.txt", 'a')
    data = "{}\n".format(i)
    f.write(str(data))
    f.close()

    i = logs.get('val_acc')
    f = open("record_valacc.txt", 'a')
    data = "{}\n".format(i)
    f.write(str(data))
    f.close()

    i = logs.get('val_loss')
    f = open("record_valloss.txt", 'a')
    data = "{}\n".format(i)
    f.write(str(data))
    f.close()



# Img_Path = "C:\\Users\\aiia\\Desktop\\inception_test\\"
# Img_Path = "C:\\Users\\aiia\\Desktop\\hanja_data_test\\"
Img_Path = "E:\\new\\"
# Img_Path = "C:\\Users\\aiia\\Desktop\\anthor-data\\"

Dp = Data_pre(img_h, img_w, chanel)
x_train, y_train, labels_val, label_dic, name = Dp.data_set_fun(Img_Path, 0)

print(y_train)
y_train = to_categorical(y_train)
print(y_train)
class_num = len(labels_val)
print(np.shape(x_train))
print(np.shape(y_train))
print(labels_val)
print(class_num)
print(label_dic)


if train_run :


    ## ?????? ?????? ??????
    f = open("record_acc.txt", 'w')
    f.close()

    f = open("record_loss.txt", 'w')
    f.close()

    f = open("record_valacc.txt", 'w')
    f.close()

    f = open("record_valloss.txt", 'w')
    f.close()

    model = InceptionResNetV2(include_top=True,
                          weights=None,
                          input_tensor=None,
                          input_shape=img_shape,
                          pooling=None,
                          classes=class_num)

    model = multi_gpu_model(model, gpus=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    record_call = [
            LambdaCallback(
                    on_epoch_end=lambda epoch, logs: record_func(epoch, logs)
            )
            ]

    history = model.fit(x_train, y_train,
                 epochs=epochs,
                 batch_size=512, # 128*128?????? 1000?????? ?????? ??????; 28*28?????? 50000?????? ?????? ??????
                 validation_split=0.1,
                 shuffle=True,
                 callbacks=record_call)

    with open('trainHistoryDict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    def plt_show_loss(history):
        plt.plot(history.histoty['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc=0)

    def plt_show_loss(history):
        plt.plot(history.histoty['acc'])
        # plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc=0)

    plt_show_loss(history)
    plt.show()

    plt_show_loss(history)
    plt.show()

else:

    model = load_model(svae_path + '%d_AE_model.h5' % epochs)
    print('model load ??????::')
    model = multi_gpu_model(model, gpus=4)
    predict = model.predict(x_train)
    print('np.shape(predict) : ', np.shape(predict))

    f = open("class_sort2.txt", 'r', encoding="UTF-8")
    line = f.readline()
    dic_line = eval(line)
    f.close()

    # ??????????????? ????????? ????????????

    classcount = 0
    # Img_Path = "C:\\Users\\aiia\\Desktop\\210701_hanja_zip\\ITKC_MO_0628A_A310\\"
    # Img_Path = 'C:\\Users\\aiia\\Desktop\\hanja_data_test\\'  # C:\Users\aiia\Desktop\hanja_data_test
    # save_path = 'D:\\error_img_save\\'
    files = os.listdir(Img_Path)

    for i, file in enumerate(files):
        # for i in range(len(x_train)):
        simi = np.max(predict[i])
        simi = simi * 10000
        if simi == 10000:
            simi_fin = '10000'
        else:
            simi = trunc(simi)
            simi_fin = '0' + str(simi)
        file = files[i]
        rename = file.split('.')
        # f_name = str(rename[0]) + '_' + str(list(dic_line)[np.argmax(predict[i])]) + '_' + str(simi_fin) + '.jpg'
        f_name = str(simi_fin)+ '_' +str(rename[0]) + '_' + str(list(dic_line)[np.argmax(predict[i])]) +'_000'  + '.jpg'
        # print(f_name)
        shutil.move(Img_Path + file, Img_Path + f_name) ## ???????????? ?????? ????????????

        # print('class : %s  pre : %s  acc : %s' %(str(label_name), str(list(dic_line)[np.argmax(predict[i])]), str(simi_fin))) # ????????? ????????????
        print('Num : %s  pre : %s  acc : %s' % (str(i), str(list(dic_line)[np.argmax(predict[i])]), str(simi_fin)))  # ????????? ????????????
        # print(simi_fin) # ????????? ???????????? ??????

    #     if str(label_name) == str(list(dic_line)[np.argmax(predict[i])]) :
    #         classcount += 1
    #     else:
    #         print('fales')
    #         # shutil.copy(Img_Path + file, save_path + f_name)  ## ???????????? ?????? ????????????
    #
    # print('acc count :', classcount)
    # print('acc rate :', classcount/np.shape(y_train)[0]*100)
    # max_val = max(predict[0])
    # print('max_val :', max_val)



