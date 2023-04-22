from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm
import cv2
import numpy as np
import math
import os
import cv2
import numpy as np


def wuniform(z, zmin, zmax):
    return np.where(np.logical_and(z >= zmin, z <= zmax), 1, 0)

def wtent(z, zmin, zmax):
    mask = np.logical_and(z >= zmin, z <= zmax)
    f = np.minimum(z, 1 - z)
    return np.where(mask, f, 0)

def wGaussian(z, zmin, zmax):
    mask = np.logical_and(z >= zmin, z <= zmax)
    f = np.exp(-4 * (z - 0.5)**2 / 0.52)
    return np.where(mask, f, 0)

def wphoton(z, zmin, zmax, tk):
    mask = np.logical_and(z >= zmin, z <= zmax)
    return np.where(mask, tk, 0)

# 读取LDR图像
ldr_list = []
for i in range(16):
    filename = f'data\\tiffdata\\exposure{i+1}.tiff'
    img = cv2.imread(filename)
    ldr_list.append(img)

# 将LDR图像转换为线性域
lin_list = []
for img in ldr_list:
    lin_img = cv2.normalize(img.astype('float32'), None, 0.0, 255, cv2.NORM_MINMAX)
    lin_list.append(lin_img)
lin_list=np.array(lin_list)
print(lin_list.shape)
# 计算每个像素的加权平均

hdr_img = np.zeros_like(lin_list[0])
for c in range(hdr_img.shape[2]):
    for i in range(hdr_img.shape[0]):
        for j in range(hdr_img.shape[1]):
            pixel_values = [lin_list[k][i,j,c] for k in range(16)]
            hdr_img[i,j,c] = np.dot(pixel_values, w)

# 将HDR图像转换为0-255范围内的像素值
hdr_img = cv2.normalize(hdr_img, None, 0, 255, cv2.NORM_MINMAX)

# 保存结果
cv2.imwrite('result.hdr', hdr_img)
