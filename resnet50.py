from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications import ResNet50
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

train_run = 1 # 1: train, 0: test

svae_path = "./resnet50_save_model/"

img_h, img_w, chanel = 128, 128, 1
img_shape = (img_h,img_w,chanel)
epochs = 100


## 기록 파일 리셋
f = open("record_acc.txt", 'w')
f.close()

f = open("record_loss.txt", 'w')
f.close()

f = open("record_valacc.txt", 'w')
f.close()

f = open("record_valloss.txt", 'w')
f.close()

def record_func(epoch, logs):
    if epoch % 20 == 0:
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
Img_Path = "C:\\Users\\aiia\\Desktop\\hanja_train\\"
Dp = Data_pre(img_h, img_w, chanel)
x_train, y_train, labels_val, label_dic = Dp.data_set_fun(Img_Path, 0)

# print(y_train)
y_train = to_categorical(y_train)
# print(y_train)
class_num = len(labels_val)
# print(np.shape(x_train))
# print(np.shape(y_train))
# print(labels_val)
# print(class_num)
# print(label_dic)

if train_run :
    model = ResNet50(include_top=True,
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
                 batch_size=512, # 128*128기준 1000배치 이하 가능; 28*28기준 50000배치 이하 가능
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
    predict = model.predict(x_train)
    print('np.shape(predict) : ', np.shape(predict))

    f = open("class_sort.txt", 'r')
    line = f.readline()
    dic_line = eval(line)
    f.close()

    ## 파일이름에 유사도 추가하기
    files = os.listdir(Img_Path)
    for i, file in enumerate(files):
        rename = file.split('.')
        simi = np.max(predict[i])
        simi = simi * 10000
        if simi == 10000:
            simi_fin = '10000'
        else:
            simi = trunc(simi)
            simi_fin = '0' + str(simi)

        # f_name = str(rename[0]) + '_' + simi_fin + '.jpg'
        f_name = str(rename[0]) + '_' + 'exc' + '.jpg'
        # print(f_name)
        # shutil.move(Img_Path + file, Img_Path + f_name)

        predic_class = str(list(dic_line)[np.argmax(predict[i])])
        print('class : %s  acc : %s' %(predic_class, str(simi_fin))) # 예측한 라벨보기
        # print(simi_fin) # 모델이 확신하는 정도
        # if predic_class != 'U9593' :
        #     shutil.move(Img_Path + file, Img_Path + f_name)

    max_val = max(predict[0])
    print('max_val :', max_val)



