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

def get_model(num_classes):
    # 加載預訓練模型
    googlenet = models.googlenet(pretrained=True)
    # 把vgg的classifier 替换成自己设置的classifier
    net = nn.Linear(googlenet.fc.in_features, num_classes) 
    net.apply(init_normal)   
    googlenet.fc = net
    return googlenet

if __name__ == "__main__":
    get_model(5)