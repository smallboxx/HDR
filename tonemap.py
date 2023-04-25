from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm,xyY_to_XYZ,XYZ_to_RGB_linear
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import cv2

def XYZ2xyY(XYZ):
    X=XYZ[:,:,0]
    Y=XYZ[:,:,1]
    Z=XYZ[:,:,2]
    x=X/(X+Y+Z)
    y=Y/(X+Y+Z)
    Y=Y
    return x,y,Y

def photographic_tonemapping(Y,value, b=0.15, k=0.18,delta=1e-9):
    Y=Y-value
    # Calculate the geometric mean (Im_HDR)
    Im_HDR = np.exp(np.mean(np.log(Y+delta)))

    # Apply the scaling factor (K)
    I_tilde_HDR = k * Y / Im_HDR

    # Calculate I_white
    I_white = b * np.max(I_tilde_HDR)

    # Perform the tonemapping operation
    I_tilde_TM = I_tilde_HDR / (1 + I_tilde_HDR / I_white**2)

    # Normalize the tonemapped image
    Y_tonemap = (I_tilde_TM / (1 + I_tilde_HDR))
    Y_tonemap = Y_tonemap
    return Y_tonemap

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

    # 将R, G, B通道数据堆叠到一个3D数组中，以便更容易地处理RGB数据
    rgb_data = np.stack((r, g, b), axis=-1)

    return rgb_data


# Load a linear HDR image in OpenEXR format
hdr_image = read_exr_to_rgb('correct_HDR.exr')

XYZ = lRGB2XYZ(hdr_image)
X=XYZ[:,:,0]
Y=XYZ[:,:,1]
Z=XYZ[:,:,2]
# plt.imshow(hdr_image)
# plt.title('RGB Image')
# plt.show()
# Apply photographic tonemapping
value = np.min(Y)
TONEMAP_Y = photographic_tonemapping(Y,value)
TONEMAP_XYZ = np.dstack([X,TONEMAP_Y,Z])
TONEMAP_RGB = XYZ2lRGB(TONEMAP_XYZ)

plt.imshow(TONEMAP_RGB)
plt.title('RGB Image')
plt.show()
# Save the tonemapped image as a PNG or JPEG
writeEXR('TONEMAP_HDR.exr',TONEMAP_RGB)