import os
import cv2
import numpy as np

from PIL import Image

img_path              = os.path.join('thesis', 'img', 'data-augmentation')
original_path         = os.path.join(img_path, 'original.jpg')
scale_distortion_path = os.path.join(img_path, 'scale-distortion.jpg')
filp_path             = os.path.join(img_path, 'filp.jpg')
rotate_path           = os.path.join(img_path, 'rotate.jpg')
hue_path              = os.path.join(img_path, 'hue.jpg')
sat_path              = os.path.join(img_path, 'sat.jpg')
val_path              = os.path.join(img_path, 'val.jpg')
gauss_path            = os.path.join(img_path, 'gauss.jpg')

original = Image.open(original_path)

#------------------------------#
#   获得图像的高宽与目标高宽
#------------------------------#
iw, ih  = 1200, 1920
h, w    = 1200, 1920


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def geometric_transformation():
    jitter=.3
    
    #------------------------------------------#
    #   对图像进行缩放并且进行长和宽的扭曲
    #------------------------------------------#
    new_ar = iw/ih * rand(1-jitter,1+jitter) / rand(1-jitter,1+jitter)
    scale = rand(0.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    temp = original.resize((nw,nh), Image.BICUBIC)
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    scale_distortion = Image.new('RGB', (w,h), (128,128,128))
    scale_distortion.paste(temp, (dx, dy))
    scale_distortion.save(scale_distortion_path)

    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = original.transpose(Image.FLIP_LEFT_RIGHT)
    filp.save(filp_path)

    #------------------------------------------#
    #   旋转
    #------------------------------------------#
    center      = (w // 2, h // 2)
    rotation    = np.random.randint(-10, 11)
    M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
    rotate_data = cv2.warpAffine(np.array(original, np.uint8), M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
    rotate      = Image.fromarray(rotate_data)
    rotate.save(rotate_path)

def color_transformation():
    hue = 0.1
    sat = 0.7 
    val = 0.3
    original_data = np.array(original)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r = np.random.uniform(-1, -0.9, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val   = cv2.split(cv2.cvtColor(original_data, cv2.COLOR_RGB2HSV))
    dtype           = original_data.dtype
    #---------------------------------#
    #   应用变换
    #---------------------------------#
    x       = np.arange(0, 256, dtype=r.dtype)

    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    hue_data = cv2.merge((cv2.LUT(hue, lut_hue), sat, val))
    hue_data = cv2.cvtColor(hue_data, cv2.COLOR_HSV2RGB)

    sat_data = cv2.merge((hue, cv2.LUT(sat, lut_sat), val))
    sat_data = cv2.cvtColor(sat_data, cv2.COLOR_HSV2RGB)

    val_data = cv2.merge((hue, sat, cv2.LUT(val, lut_val)))
    val_data = cv2.cvtColor(val_data, cv2.COLOR_HSV2RGB)

    hue_img = Image.fromarray(hue_data)
    sat_img = Image.fromarray(sat_data)
    val_img = Image.fromarray(val_data)
    
    hue_img.save(hue_path)
    sat_img.save(sat_path)
    val_img.save(val_path)

def effect_transformation():

    original_data      = np.array(original, np.uint8)

    #------------------------------------------#
    #   高斯模糊
    #------------------------------------------#
    gauss_data = cv2.GaussianBlur(original_data, (7, 7), 100, 100)
    gauss = Image.fromarray(gauss_data)
    gauss.save(gauss_path)


if __name__ == '__main__':
    geometric_transformation()
    color_transformation()
    effect_transformation()
    