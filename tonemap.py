from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm,xyY_to_XYZ
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

def photographic_tonemapping(Y, b=0.1, k=0.18, delta=1e-9):
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


# # Load a linear HDR image in OpenEXR format
# hdr_image = read_exr_to_rgb('correct_HDR.exr')

# XYZ = lRGB2XYZ(hdr_image)

# # plt.imshow(hdr_image)
# # plt.title('RGB Image')
# # plt.show()
# # Apply photographic tonemapping

# x,y,Y = XYZ2xyY(XYZ) 
# Y_tonemap = photographic_tonemapping(Y)

# X0, Y0, Z0 = xyY_to_XYZ(x,y,Y_tonemap)
# TONEMAP_XYZ = np.dstack((X0, Y0, Z0))
# TONEMAP_RGB = XYZ2lRGB(TONEMAP_XYZ)
# print(TONEMAP_RGB.shape,np.max(TONEMAP_RGB))

# # Save the tonemapped image as a PNG or JPEG
# writeEXR('TONEMAP_HDR.exr',TONEMAP_RGB)

# Load a linear HDR image in OpenEXR format
hdr_image = cv2.imread('correct_HDR.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# Apply photographic tonemapping
tonemapped_image = photographic_tonemapping(hdr_image)

# Scale the tonemapped image to the range [0, 255]
tonemapped_image_8bit = np.clip(tonemapped_image * 255, 0, 255).astype(np.uint8)

# Save the tonemapped image as a PNG or JPEG
cv2.imwrite('tonemapped_image.png', tonemapped_image_8bit)