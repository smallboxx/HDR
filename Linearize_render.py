import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import math
from concurrent.futures import ThreadPoolExecutor

import sys

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def apply_g_function(pixel, g_function):
    weight = np.exp(g_function[pixel])
    linear_pixel = int(pixel * weight)
    linear_pixel = max(0, min(linear_pixel, 255))
    return linear_pixel

def process_pixel(args):
    return apply_g_function(*args)


def process_image(image, g_function, num_threads=256, idx=1):
    h,w,k=image.shape
    processed_image = np.zeros_like(image)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for c in range(k):
            for i in range(h):
                for j in range(w):
                        pixel = image[i,j,c]
                        future = executor.submit(process_pixel, (pixel, g_function))
                        processed_image[i, j, c] = future.result()
                print_progress_bar(c*h+i+1,h*k,prefix='Progress:', suffix='Complete merge hdr', length=100)
    return processed_image

def load_images(image_files):
    imgs=[]
    for index in image_files:
        img =cv2.imread(index)
        imgs.append(img)
    return imgs

g = np.load("g_function.npy")
# Loading exposure images into a list
image_files = ['data\\door_stack\\exposure1.jpg', 'data\\door_stack\\exposure2.jpg', 'data\\door_stack\\exposure3.jpg', 'data\\door_stack\\exposure4.jpg',
               'data\\door_stack\\exposure5.jpg', 'data\\door_stack\\exposure6.jpg', 'data\\door_stack\\exposure7.jpg', 'data\\door_stack\\exposure8.jpg',
               'data\\door_stack\\exposure9.jpg', 'data\\door_stack\\exposure10.jpg','data\\door_stack\\exposure11.jpg','data\\door_stack\\exposure12.jpg',
               'data\\door_stack\\exposure13.jpg','data\\door_stack\\exposure14.jpg','data\\door_stack\\exposure15.jpg','data\\door_stack\\exposure16.jpg']

images=load_images(image_files=image_files)
# Apply the g function to each image
g_images = []
for id in range(16):
    # 使用g函数转换为线性图像
    filename = "linear_image_" + str(id+1) + ".jpg"
    linear_tiff_img = process_image(images[id],g_function=g,num_threads=256,idx=id)
    cv2.imwrite(filename, linear_tiff_img)
