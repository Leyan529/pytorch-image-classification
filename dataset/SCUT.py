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
from sklearn.utils import shuffle
import shutil
from tqdm import tqdm

root = 'D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\SCUT-FBP5500_v2\\'

def train_img_path(id):
    return os.path.join(root, 'Images\\%s' % (id)) 
def train_cls_path(id, cls):
    cls_path = os.path.join(root, 'train_images', '%s\\%s'%(int(cls),id)) 
    return cls_path    

def test_cls_path(id, cls):
    cls_path = os.path.join(root, 'test_images', '%s\\%s'%(int(cls),id)) 
    return cls_path 

def getDataset(processed):
    df = pd.read_excel(os.path.join(root, "All_Ratings.xlsx"), sheet_name="ALL") 
    df = df.groupby('Filename').mean().reset_index()
    df = df.round(0)
    df['Rating'] = df['Rating'].apply(lambda x : int(x)-1)
    df = shuffle(df)

    train_df = df[:int(len(df) * 0.8)]
    test_df = df[int(len(df) * 0.8):]

    if not processed:
        # ------------------------------------------------------------------------------------------       
        train_df['image_path'] = train_df['Filename'].apply(train_img_path)
        train_df['cls_path'] = train_df.apply(lambda x : train_cls_path(x['Filename'],x['Rating']),axis=1)
        # train_df.iloc[0]['image_path']
        # train_df.iloc[0]['cls_path']
        # # ------------------------------------------------------------------------------------------
        # test_df['image_path'] = test_df['ID'].apply(test_img_path)
        # test_df['cls_path'] = test_df.apply(lambda x : test_cls_path(x['ID']),axis=1)

        test_df['image_path'] = test_df['Filename'].apply(train_img_path)
        test_df['cls_path'] = test_df.apply(lambda x : test_cls_path(x['Filename'],x['Rating']),axis=1)
        # test_df.iloc[0]['image_path']
        # test_df.iloc[0]['cls_path']
        # ------------------------------------------------------------------------------------------   
        with tqdm(total=len(train_df)) as pbar:
            for i in range(len(train_df)):
                series = train_df.iloc[i]
                source, dest = series['image_path'], series['cls_path']
                shutil.move(source,dest)
                pbar.update(1)
                
        with tqdm(total=len(test_df)) as pbar:
            for i in range(len(test_df)):
                series = test_df.iloc[i]
                source, dest = series['image_path'], series['cls_path']
                shutil.move(source,dest)
                pbar.update(1)     

    #加上transforms
    transform=transforms.Compose([
        transforms.Resize((512,512), interpolation=3), #缩放图片，保持长宽比不变，最短边的长为224像素,
    #     transforms.CenterCrop(512), #从中间切出 224*224的图片    
    #     transforms.RandomResizedCrop(512),
    #     transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = ImageFolder(os.path.join(root, 'train_images'),transform=transform)
    test_dataset = ImageFolder(os.path.join(root, 'test_images'),transform=transform)   
    num_classes = len(train_dataset.classes)
    return train_dataset, test_dataset, num_classes   # train, val


if __name__ == '__main__':    
    train, val = getDataset(processed=True)
    train.dataset.classes
    #输出第0张图片的大小
    print(train[0][0].size())