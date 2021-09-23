from PIL import Image
import cv2
import numpy as np
import os

class Data_pre():
    def __init__(self, img_h, img_w, chanel):
        self.img_h = img_h
        self.img_w = img_w
        self.chanel = chanel

    def dataset(self, images):
        # data = pd.read_csv(PATH, header=None)
        # images = data.iloc[:, :].values
        images = images.astype(np.float)
        images = images.reshape(self.img_h,self.img_w,self.chanel)
        # print(images)
        #print('images', images)
        images = np.multiply(images, 1.0 / 255.0)
        images = np.multiply(images, 1.0 / 255.0)

        return images


    def data_set_fun(self, path, set_size, num=0):
        global labels_val

        # train = True
        # filename_list = os.listdir(path)
        # if num == 0 :
        #     np.random.shuffle(filename_list)
        #
        # filename_list =filename_list[:250000]
        #
        # if set_size == 0:
        #     set_size = len(filename_list)
        #     train = False
        #
        # X_set = np.empty((set_size, self.img_h,self.img_w,self.chanel), dtype=np.float32)
        # Y_noise = np.empty((set_size, 2), dtype=np.float32)
        # Y_set = np.empty((set_size), dtype=np.float32)
        # name = []
        #
        # result = dict()

        train = True
        filename_list = os.listdir(path)
        if set_size == 0:
            set_size = len(filename_list)
            train = False

        X_set = np.empty((set_size, self.img_h, self.img_w, self.chanel), dtype=np.float32)
        # Y_noise = np.empty((set_size, 2), dtype=np.float32)
        # Y_set = np.empty((set_size), dtype=np.float32)
        name = []

        # np.random.seed(1234)
        # np.random.shuffle(filename_list)
        result = dict()

        for i, filename in enumerate(filename_list): ## (주의)이름순으로 들어간다
            if i >= set_size:
                break
            # name.append(filename)
            label = filename.split('.')[0]
            # print(label)
            # label = label.split('_')[2]
            # label_noise = label.split('_')[0]
            label = label.split('_')[-2]

            # if label_noise == '0':
            #     Y_noise[i] = (1, 0)
            # else : Y_noise[i] = (0,1)
            # print(label)
            result[label] = result.setdefault(label, 0) + 1
            # print("label",label)

            # name.append(filename)
            name.append(label)
            # name.append(label_noise)

            # Y_set[i] = label
            # Y_set[i] = filename

            file_path = os.path.join(path, filename)
            img = Image.open(file_path)
            img = img.convert('L')  # convert image to black and white = 1, gray = L, color = RGB
            imgarray = np.array(img)
            ret, imgarray = cv2.threshold(imgarray, 200, 255, cv2.THRESH_BINARY)
            imgarray = imgarray.flatten()
            # print(imgarray)
            images = self.dataset(imgarray)

            X_set[i] = images

        # if num == 0:
        #     labels_val = list(set(name))
        #     labels_val.sort()
        labels_val = list(set(name))
        labels_val.sort()
        # if train:
        #    return X_set, Y_set, result
        Y_set = self.index_label(name)
        return X_set, Y_set, labels_val, result, name


    def dence_to_one_hot(self, labels_dence, num_classes):
        # print(labels_dence)
        num_labes = labels_dence.shape[0]
        # print(num_labes)
        index_offset = np.arange(num_labes) * num_classes
        # print(index_offset)
        labels_one_hot = np.zeros((num_labes, num_classes))
        # print(labels_dence.ravel())
        labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1  # flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
        return labels_one_hot


    def index_label(self, label):
        # print(label)
        list = []
        for j in range(len(label)):
            for i in range(len(labels_val)):
                if label[j] == labels_val[i]:
                    list.append(i)
                    break
        return np.asarray(list)