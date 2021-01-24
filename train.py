import numpy as np
import argparse
import torch

from dataset import AOI
from torch.utils.data import DataLoader
from model import cnn, resnet, pre_resnet, pre_shufflenet, pre_squeezenet, pre_mobilenet, pre_googlenet, pre_densenet, pre_inception, pre_vgg, pre_alexnet

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
from torchsummary import summary


if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', default='AOI', type=str, help='training dataset, AOI, ')
    parser.add_argument('--model', type=str, default="pre_alexnet", help='cnn/resnet/pre_resnet/pre_squeezenet/pre_mobilenet/pre_shufflenet/pre_googlenet/pre_inception/pre_vgg/pre_alexnet') 
    parser.add_argument('--start_epoch', default=1, type=int, help='starting epoch')
    parser.add_argument('--epochs', default=20
    , type=int, help='number of iterations to train' )
    parser.add_argument('--save_dir', default=None, nargs=argparse.REMAINDER, help='directory to save models' )
    parser.add_argument('--num_workers', default=0, type=int, help='number of worker to load data' )

    # cuda
    parser.add_argument('--cuda', default=True, type=bool, help='whether use CUDA')
    # multiple GPUs
    parser.add_argument('--mGPUs', default=False, help='whether use multiple GPUs')
    parser.add_argument('--gpu_ids', default='0', type=str, help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)' )
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    # batch size
    parser.add_argument('--batch_size', help='batch_size', default=4, type=int)

    # config optimization
    parser.add_argument('--optimizer', help='training optimizer', default='sgd', type=str)
    parser.add_argument('--lr', help='starting learning rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', help='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', help='step to do learning rate decay, uint is epoch', default=50, type=int)
    parser.add_argument('--lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    parser.add_argument('--loss', help='Segmentation Losses Choices: [ce or focal]', default='ce', type=str)
    

    # set training session
    parser.add_argument('--s', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument('--r', help='resume checkpoint or not', default=False, type=bool)
    parser.add_argument('--checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', help='checkpoint to load model', default=0, type=int)

    # log and display
    parser.add_argument('--use_tfboard', help='whether use tensorflow tensorboard', default=True, type=bool)

    # configure validation
    parser.add_argument('--no_val', help='not do validation', default=True, type=bool)
    parser.add_argument('--eval_interval', help='iterval to do evaluate', default=1, type=int)

    parser.add_argument('--checkname', help='checkname', default=None, type=str)


    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    train_dataset, val_dataset, test_dataset, target = None, None, None, None
    BATCH_SIZE = opt.batch_size
    num_classes = None
    # Define Dataloader
    if opt.dataset == 'AOI':
        train_dataset, val_dataset, test_dataset, target, num_classes = AOI.getDataset(processed=True)

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

    pred_loader = DataLoader(
        dataset=test_dataset,# TensorDataset类型数据集
        batch_size=3,# mini batch size
        num_workers=5# 加载数据的进程个数
    )

    # Define network
    model = None
    if opt.model == "cnn":
        model = cnn.CNN(num_classes=num_classes)
        opt.checkname = opt.model
    elif opt.model == "resnet":
        model = resnet.ResNet(num_classes=num_classes)
        opt.checkname = opt.model
    elif opt.model == "pre_resnet":
        model, layer = pre_resnet.get_model(num_classes=num_classes, layer=152)
        opt.checkname = opt.model + "_{}".format(layer)
    elif opt.model == "pre_vgg":
        model, layer = pre_vgg.get_model(num_classes=num_classes, layer=16) 
        opt.checkname = opt.model + "_{}".format(layer)
    elif opt.model == "pre_alexnet":
        model = pre_alexnet.get_model(num_classes=num_classes) # 85
        opt.checkname = opt.model
    elif opt.model == "pre_squeezenet":
        model = pre_squeezenet.get_model(num_classes=num_classes) # convergence too slow
        opt.checkname = opt.model
    elif opt.model == "pre_mobilenet":
        model = pre_mobilenet.get_model(num_classes=num_classes) # 85
        opt.checkname = opt.model
    elif opt.model == "pre_shufflenet":
        model = pre_shufflenet.get_model(num_classes=num_classes) # 97
        opt.checkname = opt.model
    elif opt.model == "pre_googlenet":
        model = pre_googlenet.get_model(num_classes=num_classes) # 80
        opt.checkname = opt.model
    elif opt.model == "pre_densenet":
        model, layer = pre_densenet.get_model(num_classes=num_classes, layer=161) 
        opt.checkname = opt.model + "_{}".format(layer)
    elif opt.model == "pre_inception":
        model = pre_inception.get_model(num_classes=num_classes) # 79
        opt.checkname = opt.model

    print(opt.checkname)
    summary(model.cuda(), (3, 224, 224))
    # multiple mGPUs
    print("device : ", device)
    if device.type == 'cpu':
        model = torch.nn.DataParallel(model)
    else:
        num_gpus = [i for i in range(opt.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=num_gpus).cuda()    

    # Define Criterion  
    criterion = nn.CrossEntropyLoss() 

    # Define Optimizer
    optimizer = None
    if opt.optimizer == 'adam':
        opt.lr = opt.lr * 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, momentum=0, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)


    # Define Saver
    saver = Saver(opt)
    saver.save_experiment_config(model.module)

    '''Train'''
    # Resuming checkpoint
    best_pred = 0.0
    lr_stage = [68, 93]
    lr_staget_ind = 0 
    train_loss = 0.0
    model.train()

    total_step = len(train_loader)
    total_train_step = opt.epochs * total_step

    print('Starting Epoch:', opt.start_epoch)
    print('Total Epoches:', opt.epochs)

    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(saver, patience=3, verbose=True)


    for epoch in range(opt.epochs):
        with tqdm(total=total_step, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='img') as pbar:
            since = time.time()
            running_loss = 0.0
            running_correct = 0.0
            for step, (images, labels) in enumerate(train_loader):
                images = get_variable(images)
                labels = get_variable(labels)

                model.train()
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                
                # calculate the loss/acc later
                running_loss += loss.item()
                running_correct += (labels==predicted).sum().item()

            epoch_duration = time.time()-since
            epoch_loss = running_loss/len(train_loader)
            epoch_acc = 100/BATCH_SIZE*running_correct/len(train_loader)
            print("\nEpoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
            
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)

            # switch the model to eval mode to evaluate on test data
            '''EVAL'''
            model.eval()
            test_acc, test_loss = eval_model(model, val_loader, criterion)
            print("eval_loss: %.4f, eval_acc: %.4f\n" % (test_loss, test_acc))           
            test_accuracies.append(test_acc)
            
            # re-set the model to train mode after validating
            model.train()
            scheduler.step(test_acc)
            since = time.time()

            early_stopping(model, optimizer, epoch, test_acc) # update patience
            if early_stopping.early_stop:
                print("Early stopping epoch %s"%(epoch))
                break

    print('Finished Training')

    '''TEST'''
    print('Start Predict dataframe')
    pred_model(model, pred_loader, target, saver.experiment_dir, opt.checkname)