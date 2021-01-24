import pandas as pd
import os

import time
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim

root = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\AOI自動光學瑕疵檢測\\Stage1\\'
train_df = pd.read_csv(os.path.join(root, 'train.csv'))
test_df = pd.read_csv(os.path.join(root, 'test.csv'))

def train_img_path(id):
    return os.path.join(root, '%s_images\\%s' % ('train', id))
def train_cls_path(id, cls):
    cls_path = os.path.join(root, '%s_images' % 'train', '%s\\%s'%(cls,id) )
    # image_path = os.path.join(root, '%s_images' % 'train', '%s'%(id) )
    return cls_path      

def test_img_path(id):
    return os.path.join(root, '%s_images\\%s' % ('test', id))
def test_cls_path(id):
    cls_path = os.path.join(root, '%s_images' % 'test', '%s\\%s'%(0,id) ) # fix to default cls-0
    # image_path = os.path.join(root, '%s_images' % 'test', '%s'%(id) )
    return cls_path    

def getDataset(processed):
    train_df = pd.read_csv(os.path.join(root, 'train.csv'))
    test_df = pd.read_csv(os.path.join(root, 'test.csv'))
    if not processed:
        # train_df = pd.read_csv(os.path.join(root, 'train.csv'))
        # test_df = pd.read_csv(os.path.join(root, 'test.csv'))
        # ------------------------------------------------------------------------------------------
        train_df['image_path'] = train_df['ID'].apply(train_img_path)
        train_df['cls_path'] = train_df.apply(lambda x : train_cls_path(x['ID'],x['Label']),axis=1)
        # train_df.iloc[0]['image_path']
        # train_df.iloc[0]['cls_path']
        # ------------------------------------------------------------------------------------------
        test_df['image_path'] = test_df['ID'].apply(test_img_path)
        test_df['cls_path'] = test_df.apply(lambda x : test_cls_path(x['ID']),axis=1)
        # test_df.iloc[0]['image_path']
        # test_df.iloc[0]['cls_path']
        # ------------------------------------------------------------------------------------------        

    #加上transforms
    transform=transforms.Compose([
        transforms.Resize((512,512), interpolation=3), #缩放图片，保持长宽比不变，最短边的长为224像素,
    #     transforms.CenterCrop(512), #从中间切出 224*224的图片    
    #     transforms.RandomResizedCrop(512),
    #     transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize([0.5], [0.5])
    ])

    full_dataset = ImageFolder(os.path.join(root, 'train_images'),transform=transform)
    val_dataset = ImageFolder(os.path.join(root, 'test_images'),transform=transform)
   
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(full_dataset, [train_size, test_size])
    num_classes = len(train_dataset.dataset.classes)
    return train_dataset, test_dataset, val_dataset, test_df, num_classes  # train, val, test


if __name__ == '__main__':    
    train, val, test = getDataset(processed=True)
    train.dataset.classes
    #输出第0张图片的大小
    print(train[0][0].size())