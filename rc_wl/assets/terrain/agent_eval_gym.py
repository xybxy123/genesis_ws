import cv2
import numpy as np
import math

img_size = 105
horizontal_scale = 0.1
vertical_scale = 0.001   #垂直缩放(单像素高度)
min_height = 0.01/0.001

# 生成一个100x100的随机噪声图像
mean_noise_hight = 0.05 #噪声平均高度 m
agent_gym = np.random.randint(0, mean_noise_hight/vertical_scale, (img_size, img_size), dtype=np.uint8)

#边缘入口
top_left = 0
color = 0
thickness = 1
for i in range(0,5):
    color +=i*5
    cv2.rectangle(agent_gym, (top_left+i,top_left+i), (img_size-i,img_size-i), color, thickness)

#滑梯 2m宽 20像素
slide_thickness = 2.0
angle_degrees = 17
max_height = math.tan(math.radians(angle_degrees))*(img_size*horizontal_scale) #现实高度 
flag = 1 #1上坡 -1下坡 0维持
keep = 2
cnt = 0
color = min_height
cv2.line(agent_gym, (0,0), (0,int(slide_thickness/horizontal_scale)), min_height, 1)
for x in range(1,img_size):
    match flag:
        case 1:
            color += (1/img_size)*max_height / vertical_scale
        case -1:
            color -= (1/img_size)*max_height / vertical_scale
        case 0:
            cnt += 1
    if cnt>keep:
        if color==255:
            flag=-1
        elif color==min_height:
            flag=1
        cnt=0
    else:
        if color>255:
            color = 255
            flag=0
        elif color < 0:
            color=min_height
            flag=0
    cv2.line(agent_gym, (x,0), (x,int(slide_thickness/horizontal_scale)), color, 1)
    
print(f"horizontal_scale:{horizontal_scale} vertical_scale:{vertical_scale}")
print(f"terrain size:{img_size*horizontal_scale} m")
# 保存图像
cv2.imwrite("./png/agent_eval_gym.png", agent_gym)
