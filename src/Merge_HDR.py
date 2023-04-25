from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm
import cv2
import numpy as np
import math
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

# progress_bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

# load img
def load_images(image_files):
    imgs=[]
    for index in image_files:
        img =cv2.imread(index)
        #img_normalized = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        imgs.append(img/255.0)
    return imgs

# pixel_process
def apply_function(pixellist, exposure_times,w_method,weigthting_method):
    pixel=0
    if weigthting_method == "linear":
        molecular   = 0
        denominator = 0
        if w_method == "uniform":
            # print("linear and uniform")
            for i in range(len(pixellist)):
                w = wuniform(pixellist[i])
                molecular  += w*(pixellist[i]/exposure_times[i])
                denominator+= w
        elif w_method == "tent":
            # print("linear and tent")
            for i in range(len(pixellist)):
                w = wtent(pixellist[i])
                molecular  += w*(pixellist[i]/exposure_times[i])
                denominator+= w
        elif w_method == "Gaussian":
            # print("linear and Gaussian")
            for i in range(len(pixellist)):
                w = wGaussian(pixellist[i])
                molecular  += w*(pixellist[i]/exposure_times[i])
                denominator+= w
        elif w_method == "photo":
            # print("linear and photo")
            for i in range(len(pixellist)):
                w = wphoton(pixellist[i],exposure_times[i])
                molecular  += w*(pixellist[i]/exposure_times[i])
                denominator+= w
        if denominator == 0:
            return pixel
        elif denominator > 0:
            pixel=molecular/denominator
            return pixel

    elif weigthting_method == "log":
        molecular   = 0
        denominator = 0
        e=1e-17
        if w_method == "uniform": 
            # print("log and uniform")
            for i in range(len(pixellist)):
                w = wuniform(pixellist[i])
                molecular  += w*(np.log(pixellist[i]+e)-np.log(exposure_times[i]+e))
                denominator+= w
        elif w_method == "tent":
            # print("log and tent")
            for i in range(len(pixellist)):
                w = wtent(pixellist[i])
                molecular  += w*(np.log(pixellist[i]+e)-np.log(exposure_times[i]+e))
                denominator+= w
        elif w_method == "Gaussian":
            # print("log and Gaussian")
            for i in range(len(pixellist)):
                w = wGaussian(pixellist[i])
                molecular  += w*(np.log(pixellist[i]+e)-np.log(exposure_times[i]+e))
                denominator+= w
        elif w_method == "photo":
            # print("log and photo")
            for i in range(len(pixellist)):
                w = wphoton(pixellist[i],exposure_times[i])
                molecular  += w*(np.log(pixellist[i]+e)-np.log(exposure_times[i]+e))
                denominator+= w
        if denominator == 0:
            return pixel
        elif denominator > 0:
            pixel=np.exp(molecular/denominator)
            return pixel
    return pixel

def process_pixel(args):
    return apply_function(*args)

# w(z)
def wuniform(z, zmin=0.05, zmax=0.95):
    return np.where(np.logical_and(z >= zmin, z <= zmax), 1, 0)

def wtent(z, zmin=0.05, zmax=0.95):
    mask = np.logical_and(z >= zmin, z <= zmax)
    f = np.minimum(z, 1 - z)
    return np.where(mask, f, 0)

def wGaussian(z, zmin=0.05, zmax=0.95):
    mask = np.logical_and(z >= zmin, z <= zmax)
    f = np.exp(-4 * (z - 0.5)**2 / 0.25)
    return np.where(mask, f, 0)

def wphoton(z, tk,zmin=0.05, zmax=0.95):
    mask = np.logical_and(z >= zmin, z <= zmax)
    return np.where(mask, tk, 0)

def Img_weigthting_merge(ldr_list,exposure_times,w_method="uniform",weigthting_method="linear",num_threads=128):
    print("get hdr img by using weigthting_method:"+weigthting_method+" and w_method:"+w_method)
    h,w,k=ldr_list[0].shape
    hdr_image = np.zeros_like(ldr_list[0])
    # calculate the weighted average of each pixel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for c in range(k):
            for i in range(h):
                for j in range(w):
                    pixellist = [ldr_list[id][i,j,c] for id in range(len(ldr_list))]
                    future = executor.submit(process_pixel, (pixellist, exposure_times,w_method,weigthting_method))
                    hdr_image[i, j, c] = future.result()
                print_progress_bar(c*h+i+1,h*k,prefix='Progress:', suffix='Complete merge hdr', length=100)

    # pixel(0,1) to pixel(0,255)
    hdr_image = cv2.normalize(hdr_image, None, 0, 255, cv2.NORM_MINMAX)

    # save
    savename=weigthting_method+"_"+w_method+"_HDRIMG.exr"
    cv2.imwrite('result.hdr', hdr_image)
    writeEXR(savename,hdr_image)

    return hdr_image

if __name__=="__main__":
    # img
    image_files = ['data\\tiffdata\\exposure1.tiff', 'data\\tiffdata\\exposure2.tiff', 'data\\tiffdata\\exposure3.tiff', 'data\\tiffdata\\exposure4.tiff',
               'data\\tiffdata\\exposure5.tiff', 'data\\tiffdata\\exposure6.tiff', 'data\\tiffdata\\exposure7.tiff', 'data\\tiffdata\\exposure8.tiff',
               'data\\tiffdata\\exposure9.tiff', 'data\\tiffdata\\exposure10.tiff','data\\tiffdata\\exposure11.tiff','data\\tiffdata\\exposure12.tiff',
               'data\\tiffdata\\exposure13.tiff','data\\tiffdata\\exposure14.tiff','data\\tiffdata\\exposure15.tiff','data\\tiffdata\\exposure16.tiff']
    images=load_images(image_files=image_files)

    # exposure time
    time_list=[]
    for i in range(16): 
        ex_time=(1/2048)*math.pow(2,i) #i={0,15}
        time_list.append(ex_time)
    exposure_times = np.array(time_list, dtype=np.float32)

    # Convert the images and exposure times to numpy arrays
    images = np.array(images)
    exposure_times = np.array(exposure_times)

    # get_hdr
    # w_method="uniform" or "tent" or "Gaussian" or "photo"
    # weigthting_method="linear" or "log"
    w_method_list=["uniform","tent","Gaussian","photo"]
    weighthing_list=["linear","log"]
    for method1 in weighthing_list:
        for method2 in w_method_list:
            if method1=="linear":
                print("skip")
            else:
                hdr=Img_weigthting_merge(ldr_list=images,exposure_times=exposure_times,w_method=method2,weigthting_method=method1)