from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm,xyY_to_XYZ,XYZ_to_RGB_linear
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

def RGB_tonemapping(r,g,b,minvalue, B=0.95, K=0.15):
    HEIGHT=r.shape[0]
    WIDTH =r.shape[1] 
    ep=1e-9
    IRM=np.sum(np.log(r+ep))
    IGM=np.sum(np.log(g+ep))
    IBM=np.sum(np.log(b+ep))
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
    r_tonemap=np.add(r_t*(1+r_t/r_white**2)/(1+r_t),minvalue)
    g_tonemap=np.add(g_t*(1+g_t/g_white**2)/(1+g_t),minvalue)
    b_tonemap=np.add(b_t*(1+b_t/b_white**2)/(1+b_t),minvalue)
    
    TONEMAP = np.dstack([r_tonemap,g_tonemap,b_tonemap])
    return TONEMAP

def XYZ_tonemapping(X,Y,Z,minvalue, B=0.95, K=0.15):
    HEIGHT=Y.shape[0]
    WIDTH =Y.shape[1] 
    ep=1e-9
    IYM=np.sum(np.log(Y+ep))
    IYM=IYM/(HEIGHT*WIDTH)
    print(IYM)
    Y_T=(K/IYM)*Y
    y_white=B*np.max(Y_T)
    y_tonemap=np.add(Y_T*(1+Y_T/y_white**2)/(1+Y_T),minvalue)
    
    TONEMAP_XYZ = np.dstack([X,y_tonemap,Z])
    TONEMAP=XYZ2lRGB(TONEMAP_XYZ)
    return TONEMAP

def read_exr_to_rgb(exr_file):
    file = OpenEXR.InputFile(exr_file)
    header = file.header()
    channels = header['channels']
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    r_str = file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    g_str = file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
    b_str = file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    r = np.frombuffer(r_str, dtype=np.float32)
    g = np.frombuffer(g_str, dtype=np.float32)
    b = np.frombuffer(b_str, dtype=np.float32)
    r.shape = (height, width)
    g.shape = (height, width)
    b.shape = (height, width)

    return r,g,b


# Load a linear HDR image in OpenEXR format
r,g,b = read_exr_to_rgb('data\\correct_exr\\correct_HDR.exr')
# use RGB for tonemapping

# minvalue=np.min([np.min(r),np.min(g),np.min(b)])
# r=np.add(r,abs(minvalue))
# g=np.add(g,abs(minvalue))
# b=np.add(b,abs(minvalue))
# BLIST=[0.95]
# KLIST=[0.15]
# for B in BLIST:
#     for K in KLIST:
#         TONEMAP_IMG = RGB_tonemapping(r,g,b,minvalue,B=B,K=K)
#         savename="data\\tonemap_exr\\"+"TONEMAP_"+"B"+str(B).split('.')[1]+"K"+str(K).split('.')[1]+".exr"
#         print(savename)
#         writeEXR(savename,TONEMAP_IMG)

# use XYZ for tonemapping
RGB=np.dstack([r,g,b])
XYZ=lRGB2XYZ(RGB)
X=XYZ[:,:,0]
Y=XYZ[:,:,1]
Z=XYZ[:,:,2]
BLIST=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
KLIST=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
minvalue=np.min(Y)
Y=np.add(Y,abs(minvalue))
for B in BLIST:
    for K in KLIST:
        TONEMAP_IMG = XYZ_tonemapping(X,Y,Z,minvalue,B=B,K=K)
        savename="data\\tonemap_exr\\"+"XYZ_"+"TONEMAP_"+"B"+str(B).split('.')[1]+"K"+str(K).split('.')[1]+".exr"
        print(savename)
        writeEXR(savename,TONEMAP_IMG)