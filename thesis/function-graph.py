import os
import numpy as np
import matplotlib.pyplot as plt

img_path = os.path.join('thesis', 'img', 'function-graph')

def crossEntropyLoss_function():
    x = np.linspace(0.000000000000000000000000001, 0.99999999999999999999999999999998, 400000)  # 在[0, 1]区间内生成400个点
    
    # 绘制y1 = -log(x), y2 = -log(1 - x)图像
    y1 = -np.log(x)
    y2 = -np.log(1 - x)
    plt.plot(x, y1, label='y = -log(x)')
    plt.plot(x, y2, label='y = -log(1 - x)')

    # 添加图例
    plt.legend()
 
    # 设置标题和标签
    plt.xlabel('x', loc='right')
    plt.ylabel('y', loc='top')

    plt.ylim((0, 6))
    plt.xlim((0, 1))
    
    plt.yticks([])
    plt.xticks([0])
    
    plt.savefig(os.path.join(img_path, 'crossEntropyLoss.png'))

def mIou_and_Dice_function():
    x = np.linspace(0.000000000000000000000000001, 0.99999999999999999999999999999998, 400000)  # 在[0, 1]区间内生成400个点
 
    # 绘制y1 =  x / (2 - x), y2 = x图像
    y1 = x / (2 - x)
    y2 = x
    plt.plot(x, y1, label='y = x / (2 - x)')
    plt.plot(x, y2, label='y = x')

 
    # 添加图例
    plt.legend()
 
    # 设置标题和标签
    plt.xlabel('x', loc='right')
    plt.ylabel('y', loc='top')
 
    plt.ylim((0, 1))
    plt.xlim((0, 1))

    plt.yticks([0, 1])
    plt.xticks([0, 1])
    
    plt.savefig(os.path.join(img_path, 'mIou-and-Dice.png'))

if __name__ == '__main__':
    # 创建图形和轴
    plt.figure(figsize=(10, 6))

    crossEntropyLoss_function()
    # mIou_and_Dice_function()


