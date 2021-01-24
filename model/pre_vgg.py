from torchvision import models
import torch.nn as nn
from torch.nn import Dropout
import torch
from torchsummary import summary
# summary(vgg16.cuda(), (3, 224, 224))


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_model(num_classes, layer, bn=False):
    # 加載預訓練模型   
    vgg = None
    if layer == 11:
        if bn: vgg = models.vgg11_bn(pretrained=True)
        else: vgg = models.vgg11(pretrained=True)   # 85      
    elif layer == 13:
        if bn: vgg = models.vgg13_bn(pretrained=True)
        else: vgg = models.vgg13(pretrained=True)  
    elif layer == 16:
        if bn: vgg = models.vgg16_bn(pretrained=True)
        else: vgg = models.vgg16(pretrained=True)  
    elif layer == 19:
        if bn: vgg = models.vgg19_bn(pretrained=True)
        else: vgg = models.vgg19(pretrained=True)  
    if bn: layer = layer + "_bn"
    
    # 把vgg的fc 替换成自己设置的classifier
    # Freeze parameters so we don't backprop through them
    for param in vgg.parameters():
        param.requires_grad = False

    net = nn.Linear(vgg.classifier[6].in_features, num_classes) 
    net.apply(init_normal)   
    vgg.classifier[6] = net   
    # summary(vgg.cuda(), (3, 224, 224))
    return vgg, layer

if __name__ == "__main__":
    get_model(5)