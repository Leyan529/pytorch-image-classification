from torchvision import models
import torch.nn as nn
import torch
from torch.nn import Dropout

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_model(num_classes):
    # 加載預訓練模型
    mobilenet = models.mobilenet_v2(pretrained=True)
    # 把vgg的classifier 替换成自己设置的classifier
    net = nn.Linear(mobilenet.classifier[1].in_features, num_classes)
    net.apply(init_normal)
    mobilenet.classifier[-1] = net
    return mobilenet

if __name__ == "__main__":
    get_model(5)