import os
import random
import matplotlib

import numpy as np
import torch
import torch.nn as nn
import scipy.signal

from PIL import Image
from matplotlib import pyplot as plt

matplotlib.use('Agg')

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    # if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
    #     return image 
    # else:
    #     image = image.convert('RGB')
    #     return image 

    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    elif len(np.shape(image)) == 3 and np.shape(image)[2] == 4:
        conv = nn.Sequential(
            # nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        ) 

        a = torch.tensor(np.array(image).transpose([2, 0, 1]).astype(np.float32) / 255.0).unsqueeze(0)
        a = conv(a).squeeze(0)

        image = Image.fromarray((a.detach().numpy().transpose([1, 2, 0]) * 255.0).astype(np.uint8))

        return image
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def loss_plot(losses, val_loss, log_dir):
        iters = range(len(losses))

        plt.figure()
        plt.plot(iters, losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

def draw_figure(x, y, path, name):
            
    plt.figure()
    plt.plot(x, y, 'red', linewidth = 2, label='train ' + name)

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.title("A " + name + " Curve")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(path, "epoch_" + name.lower() + ".png"))
    plt.cla()
    plt.close("all")