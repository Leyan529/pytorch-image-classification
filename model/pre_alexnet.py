from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch
from torchsummary import summary


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_model(num_classes):
    # 加載預訓練模型   
    alexnet = models.alexnet(pretrained=True)
    # 把alexnet的fc 替换成自己设置的classifier
    # Freeze parameters so we don't backprop through them
    for param in alexnet.parameters():
        param.requires_grad = False

    net = nn.Linear(alexnet.classifier[6].in_features, num_classes) 
    net.apply(init_normal)   
    alexnet.classifier[6] = net   
    # summary(alexnet.cuda(), (3, 224, 224))
    return alexnet

if __name__ == "__main__":
    get_model(5)