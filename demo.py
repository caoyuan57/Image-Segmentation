#!/usr/bin/env python
# coding: utf-8

# # PaddleHub抠图比赛
# 
# 本示例用DeepLabv3+模型完成一键抠图。在最新作中，作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层， 其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance，该PaddleHub Module使用百度自建数据集进行训练，可用于人像分割，支持任意大小的图片输入。在完成一键抠图之后，通过图像合成，实现扣图比赛任务
# 
# **NOTE：** 如果您在本地运行该项目示例，需要首先安装PaddleHub。如果您在线运行，直接点击“运行全部”按钮即可。
# 
# 

# **项目背景**
# 
# 通过百度这几天的培训初步了解了深度学习的基本知识以及PaddlePaddle框架、AI Studio平台的使用。特别是PaddleHub提供的预训练模型可以帮助我这种初学者实现很多实际应用。本人是芭蕾爱好者，拍摄过很多芭蕾舞照片，也梦想可以在不同的背景下拍摄出更加独特的芭蕾舞照片。因此，以此为目标，通过PaddleHub提供的图像分割预训练模型轻松实现了在月球上跳芭蕾的梦想。
# 

# ## 一、定义待抠图照片
# 
# 本项目中文件夹下ballet.jpg为待预测图片

# In[1]:


get_ipython().system('pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')


# ## 二、展示待抠图照片
# 

# In[2]:


import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

# 待抠图图片
test_img_path = ["./ballet.jpg"]
img = mpimg.imread(test_img_path[0]) 

# 展示待抠图图片
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()


# ## 三、加载预训练模型
# 
# 通过加载PaddleHub DeepLabv3+模型(deeplabv3p_xception65_humanseg)实现一键抠图

# In[3]:


import paddlehub as hub
import cv2

module = hub.Module(name="deeplabv3p_xception65_humanseg")
result = module.segmentation(images=[cv2.imread('ballet.jpg')],use_gpu=False,visualization=True)

# 抠图结果展示
res = result[0]
test_img_path = res['save_path']
img = mpimg.imread(test_img_path)
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()


# ## 四、图片合成
# 
# 将抠出的芭蕾舞蹈图片合成在月球背景图片当中

# In[4]:


from PIL import Image
import numpy as np

def blend_images(fore_image, base_image):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save("blend_res_img.jpg")

    


# ## 五、展示合成图片
# 

# In[5]:



blend_images(res['save_path'], 'universe.jpg')

# 展示合成图片
plt.figure(figsize=(10,10))
img = mpimg.imread("./blend_res_img.jpg")
plt.imshow(img) 
plt.axis('off') 
plt.show()

