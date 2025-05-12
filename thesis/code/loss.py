import torch.nn as nn

def loss(mix, label):
    ce_loss   = nn.CrossEntropyLoss()(mix, label)
    dice_loss = Dice_loss(mix, label)
    loss      = ce_loss + dice_loss
    return loss
