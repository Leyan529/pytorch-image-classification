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
    densenet = None
    if layer == 121:
        densenet = models.densenet121(pretrained=True) # 76
    elif layer == 161:
        densenet = models.densenet161(pretrained=True) # 50
    elif layer == 169:
        densenet = models.densenet169(pretrained=True)
    elif layer == 201:
        densenet = models.densenet201(pretrained=True)
    # 把vgg的classifier 替换成自己设置的classifier
    net = nn.Linear(densenet.classifier.in_features, num_classes)
    net.apply(init_normal)   
    densenet.classifier = net
    return densenet, layer

if __name__ == "__main__":
    get_model(5)