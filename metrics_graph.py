import os
from utils.utils import loss_plot, draw_figure

def draw_metrics_graph():
    loss_path = os.path.join('logs', 'depth3x3_CE-augmentation-0.9')
    metrics_path = os.path.join(loss_path, 'loss')

    loss_path     = os.path.join(metrics_path, 'epoch_loss.txt')
    mIoU_path     = os.path.join(metrics_path, 'epoch_mIoU.txt')
    mPA_path      = os.path.join(metrics_path, 'epoch_mPA.txt')
    PA_path       = os.path.join(metrics_path, 'epoch_PA.txt')
    val_loss_path = os.path.join(metrics_path, 'epoch_val_loss.txt')

    epoch_every_five = [i for i in range(0, 301, 5)]
    
    for item in [mIoU_path, mPA_path, PA_path]:
        with open(item, 'r') as file:
            name = item.replace(metrics_path, '').replace('epoch_', '').replace('.txt', '').replace('\\', '')
            metrics = [float(line.rstrip('\n')) for line in file.readlines()]
            draw_figure(epoch_every_five, metrics, metrics_path, name)
            file.close()
    
    with open(loss_path, 'r') as loss_file:
        loss = [float(line.rstrip('\n')) for line in loss_file.readlines()]
        loss_file.close()
    with open(val_loss_path, 'r') as val_loss_file:
        val_loss = [float(line.rstrip('\n')) for line in val_loss_file.readlines()]
        val_loss_file.close()
        
    loss_plot(loss, val_loss, metrics_path)

if __name__ == '__main__':
    draw_metrics_graph()
    