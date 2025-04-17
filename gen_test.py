import os
import numpy as np
from PIL import Image
import cv2
        
def rand(a=0, b=1):
        return np.random.rand() * (b - a) + a

def get_random_data(image, depth, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3):
    
    #------------------------------#
    #   获得图像的高宽与目标高宽
    #------------------------------#
    iw, ih  = image.size
    h, w    = input_shape

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
    image = image.resize((nw,nh), Image.BICUBIC)
    depth = depth.resize((nw,nh), Image.NEAREST)
    label = label.resize((nw,nh), Image.NEAREST)
        
    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    flip = rand()<.5
    if flip: 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    
    #------------------------------------------#
    #   将图像多余的部分加上灰条
    #------------------------------------------#
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_depth = Image.new('L', (w, h), (0))
    new_label = Image.new('P', (w,h))
    new_image.paste(image, (dx, dy))
    new_depth.paste(depth, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    depth = new_depth
    label = new_label

    image_data      = np.array(image, np.uint8)

    #------------------------------------------#
    #   高斯模糊
    #------------------------------------------#
    blur = rand() < 0.25
    if blur: 
        image_data = cv2.GaussianBlur(image_data, (5, 5), 0)
    
    #------------------------------------------#
    #   旋转
    #------------------------------------------#
    rotate = rand() < 0.25
    if rotate: 
        center      = (w // 2, h // 2)
        rotation    = np.random.randint(-10, 11)
        M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
        image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
        depth       = cv2.warpAffine(np.array(depth, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
        label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype           = image_data.dtype
    #---------------------------------#
    #   应用变换
    #---------------------------------#
    x       = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
    return image_data, depth, label

if __name__ == "__main__":
    dataset_path = os.path.join("datasets", "PlaneEngine")
    gen_path     = os.path.join(dataset_path, "Gen")

    if not os.path.exists(gen_path):
        os.mkdir(gen_path)

    for i in range(1, 200):    
        name = str(1 + int(rand() * 113)).zfill(3)

        image = Image.open(os.path.join(dataset_path, "Images", name + ".jpg"))
        depth = Image.open(os.path.join(dataset_path, "Depths", name + ".jpg"))
        label = Image.open(os.path.join(dataset_path, "Labels", name + ".png"))

        palette = label.getpalette()
        
        image, depth, label    = get_random_data(image, depth, label, [1200, 1920])

        depth = np.array(depth)
        depth[depth > 255] = 255

        label         = np.array(label)
        label[label >= 3] = 3
        
        image = Image.fromarray(image)
        depth = Image.fromarray(depth)
        label = Image.fromarray(label)
        label.putpalette(palette)

        image.save(os.path.join(gen_path, str(i) + "image.jpg"))
        depth.save(os.path.join(gen_path, str(i) + "depth.jpg"))
        label.save(os.path.join(gen_path, str(i) + "label.png"))
