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
    # squeezenet = models.squeezenet1_0(pretrained=True)  # 28  
    squeezenet = models.squeezenet1_1(pretrained=True)  # 28 
    net = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1,1)) 
    net.apply(init_weights)
    # 把squeezenet1的classifier替换成自己设置的classifier
    squeezenet.classifier[1] = net   
    return squeezenet

if __name__ == "__main__":
    get_model(5)