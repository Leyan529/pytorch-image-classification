import numpy as np
import argparse
import torch

from dataset import AOI, SCUT
from torch.utils.data import DataLoader
from model import cnn, resnet, pre_resnet, pre_shufflenet, pre_squeezenet, pre_mobilenet, pre_googlenet, pre_densenet, pre_inception

# from losses import SegmentationLosses
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


from utils.saver import Saver
from utils.func import get_variable, eval_model, pred_model
from utils.pytorchtools import EarlyStopping
# from utils.metrics import Evaluator

import time
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import os
from glob import glob

if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument('--dataset', default='AOI', type=str, help='training dataset, AOI, SCUT')
    parser.add_argument('--model', type=str, default="pre_googlenet", help='cnn/resnet/pre_resnet/pre_squeezenet/pre_mobilenet/pre_shufflenet/pre_googlenet/pre_inception') 
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument('--experiment_dir', help='dir of experiment', type=str, default = "run\AOI\pre_googlenet\experiment_2")  

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    train_dataset, val_dataset, test_dataset, target = None, None, None, None
    BATCH_SIZE = opt.batch_size
    pred_loader = None

    # Define Dataloader
    num_classes = None
    if opt.dataset == 'AOI':
        train_dataset, val_dataset, test_dataset, target, num_classes = AOI.getDataset(processed=True)
        pred_loader = DataLoader(
            dataset=test_dataset,# TensorDataset类型数据集
            batch_size=3,# mini batch size
            num_workers=5# 加载数据的进程个数
        )

    elif opt.dataset == 'SCUT':
        train_dataset, val_dataset, num_classes = SCUT.getDataset(processed=True)

    train_loader = DataLoader(
    dataset=train_dataset,# TensorDataset类型数据集
    batch_size=BATCH_SIZE,# mini batch size
    shuffle=True,# 设置随机洗牌
    num_workers=5# 加载数据的进程个数
    )

    val_loader = DataLoader(
        dataset=val_dataset,# TensorDataset类型数据集
        batch_size=BATCH_SIZE,# mini batch size
        shuffle=True,# 设置随机洗牌
        num_workers=5# 加载数据的进程个数
    )    

    net = None
    # Define network
    if opt.model == "cnn":
        net = cnn.CNN(num_classes=num_classes)
        opt.checkname = opt.model
    elif opt.model == "resnet":
        net = resnet.ResNet(num_classes=num_classes)
        opt.checkname = opt.model
    elif opt.model == "pre_resnet":
        net, layer = pre_resnet.get_model(num_classes=num_classes, layer=101)
        opt.checkname = opt.model + "_{}".format(layer)
    # elif opt.model == "pre_vgg":
    #     model, layer = pre_vgg.get_model(num_classes=num_classes) # nan
    #     opt.checkname = opt.model + "_{}".format(layer)
    # elif opt.model == "pre_alexnet":
    #     model = pre_alexnet.get_model(num_classes=num_classes) # nan
    #     opt.checkname = opt.model
    elif opt.model == "pre_squeezenet":
        net = pre_squeezenet.get_model(num_classes=num_classes) # 58
        opt.checkname = opt.model
    elif opt.model == "pre_mobilenet":
        net = pre_mobilenet.get_model(num_classes=num_classes) # 
        opt.checkname = opt.model
    elif opt.model == "pre_shufflenet":
        net = pre_shufflenet.get_model(num_classes=num_classes) # 
        opt.checkname = opt.model
    elif opt.model == "pre_googlenet":
        net = pre_googlenet.get_model(num_classes=num_classes) # 
        opt.checkname = opt.model
    elif opt.model == "pre_densenet":
        net = pre_densenet.get_model(num_classes=num_classes) # 
        opt.checkname = opt.model
    elif opt.model == "pre_inception":
        net = pre_inception.get_model(num_classes=num_classes) # 
        opt.checkname = opt.model

    
    # Trained model path and name
    experiment_dir = opt.experiment_dir
    model_name = glob(os.path.join(opt.experiment_dir, "*.pkl"))[0]
    load_name = os.path.join(experiment_dir, 'checkpoint.pth.tar')

    # Load save/trained model
    if not os.path.isfile(model_name):
        raise RuntimeError("=> no model found at '{}'".format(model_name))
    print('====>loading trained model from ' + model_name)
    if not os.path.isfile(load_name):
        raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
    print('====>loading trained model from ' + load_name)

    net = torch.load(model_name)
    checkpoint = torch.load(load_name)


    net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    best_pred = checkpoint['best_pred']

    if device.type == 'cpu':
        model = torch.nn.DataParallel(net)
    else:
        num_gpus = [i for i in range(opt.n_gpu)]
        model = torch.nn.DataParallel(net, device_ids=num_gpus).cuda()

    '''TEST'''
    if opt.dataset == "AOI":
        print('Start Predict dataframe')
        pred_model(model, pred_loader, target, experiment_dir, opt.checkname)

    '''FACE RANK'''
    import matplotlib.image as img 
    import cv2
    from extra.face import predict_image
    if opt.dataset == "SCUT":
        image = img.imread("extra//test.jpg")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # opencvImage 
        score = predict_image(None, image, model)
        print("Predict Score : {}".format(score))