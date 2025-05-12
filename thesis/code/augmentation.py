import os
import cv2
import numpy as np

from PIL import Image

img_path              = os.path.join('essay', 'img', 'data-augmentation')
original_path         = os.path.join(img_path, 'original.jpg')
scale_distortion_path = os.path.join(img_path, 'scale-distortion.jpg')
filp_path             = os.path.join(img_path, 'filp.jpg')
rotate_path           = os.path.join(img_path, 'rotate.jpg')
hue_path              = os.path.join(img_path, 'hue.jpg')
sat_path              = os.path.join(img_path, 'sat.jpg')
val_path              = os.path.join(img_path, 'val.jpg')
gauss_path            = os.path.join(img_path, 'gauss.jpg')

original = Image.open(original_path)

def rand(a = 0, b = 1):
    return np.random.rand() * (b - a) + a

def augmentation(input_width, input_height, output_width, output_height, image, label):

    #------------------------------------------#
    #   对图像进行缩放并且进行长和宽的扭曲
    #------------------------------------------#
    jitter       = .3
    aspect_ratio = input_width / input_height
    
    new_aspect_ratio = aspect_ratio * rand(1 - jitter,1 + jitter) / rand(1 - jitter,1 + jitter)
    scale = rand(0.25, 2)

    if new_aspect_ratio < 1:
        new_height = int(scale * output_width)
        new_width = int(new_height * new_aspect_ratio)
    else:
        new_width = int(scale * output_height)
        new_height = int(new_width / new_aspect_ratio)
    
    image = image.resize((new_width,new_height), Image.BICUBIC)
    label = label.resize((new_width,new_height), Image.NEAREST)

    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = rand() < .5
    if filp:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    
    #------------------------------------------#
    #   将图像多余的部分加上灰条
    #------------------------------------------#
    dx = int(rand(0, output_width  - new_width))
    dy = int(rand(0, output_height - new_height))
    new_image = Image.new('RGB', (output_width, output_height), (128,128,128))
    new_label = Image.new('L',   (output_width, output_height), (0))
    new_image.paste(image, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    label = new_label

    #------------------------------------------#
    #   高斯模糊
    #------------------------------------------#
    image_data = np.array(image, np.uint8)
    label_data = np.array(label, np.uint8)

    blur = rand() < 0.25
    if blur: 
        image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

    #------------------------------------------#
    #   旋转
    #------------------------------------------#
    rotate = rand() < 0.25
    if rotate: 
        center     = (output_height // 2, output_width // 2)
        rotation   = np.random.randint(-10, 11)
        M          = cv2.getRotationMatrix2D(center, -rotation, scale=1)
        image_data = cv2.warpAffine(image_data, M, (output_height, output_width), flags=cv2.INTER_CUBIC,  borderValue=(128,128,128))
        label_data = cv2.warpAffine(label_data, M, (output_height, output_width), flags=cv2.INTER_NEAREST, borderValue=(0))

    #---------------------------------#
    #   对图像进行色域变换
    #---------------------------------#
    hsv = rand < .5
    if hsv:
        hue = 0.1
        sat = 0.7 
        val = 0.3
    #---------------------------------#
    #   计算色域变换的参数
    #---------------------------------#
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype         = image_data.dtype
    #---------------------------------#
    #   应用变换
    #---------------------------------#
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
    return image_data, label_data
