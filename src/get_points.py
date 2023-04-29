import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# 加载JPEG图像
img = mpimg.imread('data\\linear_render_imgs\\linear_image_16.jpg')

# 显示图像
plt.imshow(img)


# 循环选择点
while True:
    # 每次选择一个点
    point = plt.ginput(1, show_clicks=True, timeout=0)
    if len(point) == 0:
        # 如果选择结束，退出循环
        break
    # 将所选点添加到列表中
    x, y = point[0]
    # 将所选点绘制在图像上
    plt.plot(x, y, 'ro',markersize=1)
    # 将所选点的坐标写入txt文件
    with open('other_files\\correct_points.txt', 'a') as f:
        f.write(f'{x},{y}\n')
    f.close()

plt.show()

# with open('correct_points.txt','r') as f:
#     lines=f.readlines()
#     batch_x=[]
#     batch_y=[]
#     for line in lines:
#         line=line.strip().split(',')
#         batch_x.append(int(round(float(line[0]))))
#         batch_y.append(int(round(float(line[1]))))
#         if len(batch_x)==4:
#             min_x=min(batch_x)
#             max_x=max(batch_x)
#             min_y=min(batch_y)
#             max_y=max(batch_y)
#             with open('other_files\\int_correct_points.txt','a') as nf:
#                 nf.write(f'{min_x},{max_x},{min_y},{max_y}\n')
#             nf.close()
#             batch_x.clear()
#             batch_y.clear()