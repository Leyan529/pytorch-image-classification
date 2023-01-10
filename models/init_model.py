import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataloader import DataGenerator, dataset_collate

from .architectures.alexnet import alexnet
from .architectures.squeezenet import squeezenet
from .architectures.shufflenet import shufflenetv2
from .architectures.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .architectures.resnet import resnext50_32x4d, resnext101_32x8d
from .architectures.inception import inception_v3
from .architectures.googlenet import googlenet
from .architectures.efficientnet import efficientnet
from .architectures.mobilenet import mobilenet_v2
from .architectures.senet import se_resnet50, se_resnet101, se_resnet152, se_resnet50_fc512
from .architectures.senet import se_resnext50_32x4d, se_resnext101_32x4d
from .architectures.vgg16 import vgg16
from .architectures.densenet import densenet
from .architectures.vit import vit



def get_model(net_type, input_shape, pretrained=False, output_size=2):
    if net_type == 'alexnet':
        model = alexnet(input_shape = input_shape, num_classes = output_size, pretrained = pretrained)
    elif net_type == 'squeezenet':
        model = squeezenet(input_shape = input_shape, num_classes = output_size, pretrained = pretrained)
    elif net_type == 'shufflenet':
        model = shufflenetv2(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'resnet152':
        model = resnet152(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'resnext101_32x8d':
        model = resnext101_32x8d(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'inception_v3':
        model = inception_v3(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'googlenet':
        model = googlenet(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'efficientnet':
        model = efficientnet(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'mobilenet_v2':
        model = mobilenet_v2(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'se_resnet152':
        model = se_resnet152(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'se_resnet50_fc512':
        model = se_resnet50_fc512(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'se_resnext101_32x4d':
        model = se_resnext101_32x4d(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'vgg16':
        model = vgg16(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'densenet':
        model = densenet(num_classes = output_size, pretrained = pretrained)
    elif net_type == 'vit':
        model = vit(input_shape = input_shape, num_classes = output_size, pretrained = pretrained)

    else:
        raise Exception('Unknown architecture type') 
    
    return model   

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
def init_loss(criterion_name):

    if criterion_name=='bce':
        loss = nn.BCEWithLogitsLoss()
    elif criterion_name=='cce':
        loss = nn.CrossEntropyLoss()       
    elif criterion_name == 'focal_loss':
        loss = FocalLoss()
    else:
        raise Exception('This loss function is not implemented yet.') 

    return loss 


def get_optimizer(model, opt, optimizer_type):    
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'adamw' : optim.AdamW(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), opt.Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[optimizer_type]   
    return optimizer


def generate_loader(opt):     
    train_dataset   = DataGenerator(opt.lines[:opt.num_train], opt.input_shape, True)
    val_dataset     = DataGenerator(opt.lines[opt.num_train:], opt.input_shape, False)

    # gen             = DataLoader(train_dataset, shuffle = True, batch_size = opt.batch_size, num_workers = opt.num_workers, pin_memory=True,
    #                                 drop_last=True, collate_fn=dataset_collate)
    # gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = opt.batch_size, num_workers = opt.num_workers, pin_memory=True, 
    #                             drop_last=True, collate_fn=dataset_collate)   

    batch_size      = opt.batch_size
    if opt.distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // opt.ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val