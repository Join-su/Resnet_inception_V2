import cv2
import numpy as np
import os

# 빈 이미지 크기
h = 128
w = 128

# target_img = cv2.imread('C:\\Users\\ialab\\Desktop\\labeling\\result\\007Ha\\0000001_ITKC_MO_0445A_A170_007Ha_1_.jpg', cv2.IMREAD_UNCHANGED)
# target_img_h, target_img_w, target_img_c = target_img.shape


#root_dir = 'C:\\Users\\ialab\\Desktop\\labeling\\result'
#save_dir = 'C:\\Users\\ialab\\PycharmProjects\\Resnet_hanja\\img\\'

root_dir = 'C:\\Users\\aiia\\Desktop\\anthor-data\\'
save_dir = root_dir



files = os.listdir(root_dir)
for i, img in enumerate(files):
    img_path = os.path.join(root_dir, img)
    label = img[-5]
    print('img_path : ', img_path)

    # 빈이미지 생성
    blank_img = np.zeros((h, w, 3), np.uint8)

    # 한자 이미지
    target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(np.shape(target_img))
    # h,w = np.shape(target_img)
    # target_img = np.reshape(target_img,[h,w,1])
    # print(np.shape(target_img))
    # 이미지 색 반전
    target_img = cv2.bitwise_not(target_img)

    # 이미지 정보
    blank_img_h, blank_img_w, blank_img_c = blank_img.shape
    target_img_h, target_img_w, target_img_c = target_img.shape

    print(target_img_h, target_img_w, target_img_c)

    # 빈 이미지 중심점 찾기
    x = (blank_img_h - target_img_h) // 2
    y = (blank_img_w - target_img_w) // 2

    roi = blank_img[x: x + target_img_h, y:y + target_img_w]

    # 합치기
    result = cv2.add(roi, target_img)
    np.copyto(roi, result)
    blank_img = cv2.bitwise_not(blank_img)
    img = img[:]
    cv2.imwrite(save_dir + img, blank_img)
    # cv2.imshow('test', blank_img)
    # cv2.waitKey(0)
# for i,file in enumerate(files):
#     #if (i+1)>12: continue
#     #if i < 1030 : continue
#     image_dir = os.path.join(root_dir, file)
#     img_files = os.listdir(image_dir)
#     for i,img in enumerate(img_files):
#         img_path = os.path.join(image_dir, img)
#         label = img[-5]
#         print('img_path : ', img_path)
#
#         # 빈이미지 생성
#         blank_img = np.zeros((h, w, 3), np.uint8)
#
#         # 한자 이미지
#         target_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         print(np.shape(target_img))
#         #h,w = np.shape(target_img)
#         #target_img = np.reshape(target_img,[h,w,1])
#         #print(np.shape(target_img))
#         # 이미지 색 반전
#         target_img = cv2.bitwise_not(target_img)
#
#         # 이미지 정보
#         blank_img_h, blank_img_w, blank_img_c = blank_img.shape
#         target_img_h, target_img_w, target_img_c = target_img.shape
#
#         print(target_img_h, target_img_w, target_img_c)
#
#         # 빈 이미지 중심점 찾기
#         x = (blank_img_h - target_img_h) // 2
#         y = (blank_img_w - target_img_w) // 2
#
#         roi = blank_img[x: x+target_img_h, y:y+target_img_w]
#
#         # 합치기
#         result = cv2.add(roi, target_img)
#         np.copyto(roi, result)
#         blank_img = cv2.bitwise_not(blank_img)
#         img = img[:]
#         cv2.imwrite(save_dir + img, blank_img)
#         #cv2.imshow('test', blank_img)
#         #cv2.waitKey(0)
