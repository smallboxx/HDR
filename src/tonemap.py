from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm,xyY_to_XYZ,XYZ_to_RGB_linear
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math

def gamma(T):
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i][j]<= 0.0031308:
                T[i][j]=12.92*T[i][j]
            else:
                T[i][j]=(1+0.55)*math.pow(T[i][j],1/2.4)-0.55
            print_progress_bar(i*T.shape[1]+j+1,4000*6000,prefix='Progress:', suffix='Complete merge hdr', length=100)
    return T

# progress_bar
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def photographic_tonemapping(r,g,b, B=0.85, K=0.10):
    HEIGHT=r.shape[0]
    WIDTH =r.shape[1] 
    ep=1e-9
    rmin  =np.min(r)
    gmin  =np.min(g)
    bmin  =np.min(b)
    IRM=np.sum(np.log(r-rmin+ep))
    IGM=np.sum(np.log(g-gmin+ep))
    IBM=np.sum(np.log(b-bmin+ep))
    IRM=IRM/(HEIGHT*WIDTH)
    IGM=IGM/(HEIGHT*WIDTH)
    IBM=IBM/(HEIGHT*WIDTH)
    print(IRM,IGM,IBM)
    r_t=(K/IRM)*r
    g_t=(K/IGM)*g
    b_t=(K/IBM)*b
    r_white=B*np.max(r_t)
    g_white=B*np.max(g_t)
    b_white=B*np.max(b_t)
    r_tonemap=r_t*(1+r_t/r_white**2)/(1+r_t)
    g_tonemap=g_t*(1+g_t/g_white**2)/(1+g_t)
    b_tonemap=b_t*(1+b_t/b_white**2)/(1+b_t)
    
    TONEMAP = np.dstack([r_tonemap,g_tonemap,b_tonemap])
    return TONEMAP

def read_exr_to_rgb(exr_file):
    # 打开EXR文件
    file = OpenEXR.InputFile(exr_file)

    # 获取文件头信息
    header = file.header()

    # 获取文件中的通道信息，包括颜色和空间
    channels = header['channels']
    
    # 获取图像大小
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 获取R, G, B通道的数据
    r_str = file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    g_str = file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
    b_str = file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))

    # 将数据转换为NumPy数组
    r = np.frombuffer(r_str, dtype=np.float32)
    g = np.frombuffer(g_str, dtype=np.float32)
    b = np.frombuffer(b_str, dtype=np.float32)

    # 将一维数组重塑为二维图像数组
    r.shape = (height, width)
    g.shape = (height, width)
    b.shape = (height, width)

    return r,g,b


# Load a linear HDR image in OpenEXR format
r,g,b = read_exr_to_rgb('correct_HDR.exr')

TONEMAP_IMG = photographic_tonemapping(r,g,b)
writeEXR('TONEMAP_B085_K010.exr',TONEMAP_IMG)