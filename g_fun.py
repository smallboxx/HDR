import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import math

def load_images(image_files):
    grays=[]
    for index in image_files:
        img =cv2.imread(index,cv2.IMREAD_UNCHANGED)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
    return grays

def sample_pixels(images, num_samples):
    h, w = images[0].shape
    idx = np.random.randint(0, h * w, size=num_samples)
    samples = np.zeros((num_samples, len(images), 1), dtype=np.uint8)
    for i, img in enumerate(images):
        img_flat = img.reshape(-1, 1)
        samples[:, i, :] = img_flat[idx, :]
    return samples

# Loading exposure images into a list
image_files = ["data\\rawdata\\exposure1.nef", "data\\rawdata\\exposure2.nef", 'data\\rawdata\\exposure3.nef', 'data\\rawdata\\exposure4.nef',
               'data\\rawdata\\exposure5.nef', 'data\\rawdata\\exposure6.nef', 'data\\rawdata\\exposure7.nef', 'data\\rawdata\\exposure8.nef',
               'data\\rawdata\\exposure9.nef', 'data\\rawdata\\exposure10.nef','data\\rawdata\\exposure11.nef','data\\rawdata\\exposure12.nef',
               'data\\rawdata\\exposure13.nef','data\\rawdata\\exposure14.nef','data\\rawdata\\exposure15.nef','data\\rawdata\\exposure16.nef']


images=load_images(image_files=image_files)
time_list=[]
for i in range(16): 
    ex_time=(1/2048)*math.pow(2,i) #i={0,15}
    time_list.append(ex_time)

# Convert the images and exposure times to numpy arrays
images = np.array(images, dtype=np.float32)
exposure_times = np.array(time_list, dtype=np.float32)
print(exposure_times)

# Compute the image dimensions
height, width = images[0].shape

# Choose a random subset of pixels to use for estimating g
samples = sample_pixels(images, num_samples=10000)
N, P, C = samples.shape
Z = samples.reshape(N * P, C)  # Pixel values
B = np.log(exposure_times) # Logarithm of exposure times


# Set up the A matrix for the least-squares problem
A = np.zeros((N * P, 256), dtype=np.float32)
b = np.zeros((A.shape[0], C), dtype=np.float32)
epsilon = 1e-13

# Data-fitting equations
k = 0
for i in range(N):
    for j in range(P):
        wij = 1        
        A[k, Z[i*j,0]] = wij*(Z[i*j,0]+epsilon)
        b[k] = wij * B[j]+np.log(epsilon)
        k += 1

# Solve the least-squares problem to estimate g
x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
g = x[:256]

# change strange value
temp_i=0
data_min=min(g[:20])
for i in range(10):
    if g[i]==data_min:
        temp_i=i
flag=0
while(flag<temp_i):
    g[flag]=data_min*(temp_i-flag+1)
    flag+=1
print(g)

# Plot the estimated g function
plt.plot(np.arange(256), g)
plt.title("Estimated g Function")
plt.xlabel("Pixel Value")
plt.ylabel("Log Exposure Time")
plt.savefig("g_function.png")
np.save("g_function.npy", g)




