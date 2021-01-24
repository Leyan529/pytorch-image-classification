from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_model(num_classes, layer):
    # 加載預訓練模型
    resnet = None
    if layer == 34:
        resnet = models.resnet34(pretrained=True) # acc=59
    elif layer == 50:
        resnet = models.resnet50(pretrained=True) # acc=72
    elif layer == 101:
        resnet = models.resnet101(pretrained=True) # 80
    elif layer == 152:
        resnet = models.resnet152(pretrained=True) # 73
    # 把resnet的全连接层fc 替换成自己设置的线性层nn.Linear
    # 输入维度是resnet.fc.in_features, 输出是196维
    net = nn.Linear(resnet.fc.in_features, num_classes)
    net.apply(init_normal)   
    resnet.fc = net
    return resnet, layer