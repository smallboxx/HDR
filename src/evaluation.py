from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
# load_EXR_file
file_list=['data\\exr_by_merge\\linear_Gaussian_HDRIMG.exr','data\\exr_by_merge\\linear_photo_HDRIMG.exr','data\\exr_by_merge\\linear_tent_HDRIMG.exr',
           'data\\exr_by_merge\\linear_uniform_HDRIMG.exr','data\\exr_by_merge\\log_Gaussian_HDRIMG.exr','data\\exr_by_merge\\log_photo_HDRIMG.exr',
           'data\\exr_by_merge\\log_tent_HDRIMG.exr','data\\exr_by_merge\\log_uniform_HDRIMG.exr']
y_values = []
for filename in file_list:
    exr_file = OpenEXR.InputFile(filename)

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
    lrgb = np.stack([r, g, b], axis=-1)

    # LRGB2XYZ
    Y=lRGB2XYZ(lrgb)[:,:,1]
    ptr=[]
    with open('other_files\\int_points.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip().split(',')
            ptr.append(line)
    f.close()
    color_check=[]
    for id in range(len(ptr)):
        w_min=int(ptr[id][0])
        w_max=int(ptr[id][1])
        h_min=int(ptr[id][2])
        h_max=int(ptr[id][3])
        w=w_max-w_min
        h=h_max-h_min
        pixel_num=w*h
        value=0
        for i in range(h_min,h_max):
            for j in range(w_min,w_max):
                value+=Y[i,j]
        color_check.append(np.log(value/pixel_num+1e-7))
    y_values.append(color_check)
    print(color_check)

# Plot the y_values on a graph
for i in range(len(y_values)):
    plt.plot(range(1,7),y_values[i],color=(i/16,i/17,i/18,),label="line"+str(i))

# Add labels and title to the graph
plt.title('Average Luminance Values')
plt.xlabel('Region Index')
plt.ylabel('Log Average Luminance')

# Display the graph
plt.show()