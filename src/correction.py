from src.cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
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

# load_EXR_file
exr_file = OpenEXR.InputFile('data\\exr\\fromrenderjpg_linear_Gaussian_HDRIMG.exr')

# get_window_size
dw = exr_file.header()["dataWindow"]
width = dw.max.x - dw.min.x + 1
height = dw.max.y - dw.min.y + 1

# get_channel_name
channel_list = exr_file.header()["channels"].keys()

# bulid dict 
channel_dict = {}
for channel in channel_list:
    channel_dict[channel] = np.zeros((height, width), dtype=np.float32)

# load data
for channel in channel_list:
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = exr_file.channel(channel, pixel_type)
    channel_dict[channel] = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)

# data 2 rgb
r = channel_dict["R"]
g = channel_dict["G"]
b = channel_dict["B"]

# bulid b
standard_r,standard_g,standard_b=read_colorchecker_gm()
standard_l = np.ones_like(standard_r)
STANDARD_RGB = np.dstack([standard_r,standard_g,standard_b,standard_l])
list_b = STANDARD_RGB.reshape(-1,4)

#bulid A
cal_r=np.zeros((4,6))
cal_g=np.zeros((4,6))
cal_b=np.zeros((4,6))
cal_l=np.ones((4,6))
with open('correct_points.txt','r') as f:
    lines=f.readlines()
    for i in range(len(lines)):
        row=i%4
        col=i//4
        line=lines[i].strip().split(',')
        center_point=[int(float(line[0])),int(float(line[1]))]
        num=3600 #80*80
        tempr=0
        tempg=0
        tempb=0
        for i in range(center_point[1]-30,center_point[1]+30):
            for j in range(center_point[0]-30,center_point[0]+30):
                tempr+=r[i][j]
                tempg+=g[i][j]
                tempb+=b[i][j]
        cal_r[row][col]=tempr/num
        cal_g[row][col]=tempg/num
        cal_b[row][col]=tempb/num
        if row==3 and col==0:
            patch4=[center_point[1]-30,center_point[1]+30,center_point[0]-30,center_point[0]+30]
f.close()

# # show standard sRGB
CAL_RGB= np.dstack([cal_r,cal_g, cal_b,cal_l])
A = CAL_RGB.reshape(-1,4)
x, _, _, _ =np.linalg.lstsq(A,list_b, rcond=None)
# rgb = np.dstack([standard_r,standar_g, standar_b])
# plt.imshow(rgb)
# plt.title('RGB Image')
# plt.show()
# print(x)
# use x to correct HDR
l = np.ones_like(r)
RGB=np.dstack([r,g,b,l])
TRANS_RGB=RGB.reshape(-1,4)
H = 4000
W = 6000
# print(r.shape)
# print(TRANS_RGB.shape)
# trans rgb
transformed_image_2d = np.dot(TRANS_RGB,x)
# retrans
print(transformed_image_2d[:3,:])

transformed_image = transformed_image_2d[:,:3].reshape((H, W, 3))
# white trans
w_r=0
w_g=0
w_b=0
for i in range(patch4[0],patch4[1]):
    for j in range(patch4[2],patch4[3]):
        w_r+=transformed_image[:,:,0][i][j]
        w_g+=transformed_image[:,:,1][i][j]
        w_b+=transformed_image[:,:,2][i][j]
s_r=standard_r[3][0]/(w_r/3600)
s_g=standard_g[3][0]/(w_g/3600)
s_b=standard_b[3][0]/(w_b/3600)
print(s_r,s_g,s_b)

for i in range(transformed_image.shape[0]):
    for j in range(transformed_image.shape[1]):
        transformed_image[i, j, 0] *= s_r
        transformed_image[i, j, 1] *= s_g
        transformed_image[i, j, 2] *= s_b
        print_progress_bar(i*6000+j+1,6000*4000,prefix='Progress:', suffix='Complete merge hdr', length=100)
writeEXR('correct_HDR.exr',transformed_image)
plt.imshow(transformed_image)
plt.title('RGB Image')
plt.show()