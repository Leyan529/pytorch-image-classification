import sys
sys.path.append("../")
import os
from os import getcwd
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import random
import numpy as np
#---------------------------------------------------#
#   训练自己的数据集的时候一定要注意修改classes
#   修改成自己数据集所区分的种类
#   
#   种类顺序需要和训练时用到的model_data下的txt一样
#---------------------------------------------------#

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


sets    = ["train", "test"]

classes, _ = get_classes("model_data/SCUT_classes.txt")

traintest = 0.1

text_to_id = {}
id_to_text = {}

def train_img_path(root, id):
    return os.path.join(root, 'Images/%s' % (id)) 
def train_cls_path(root, id, cls):
    return os.path.join(root, 'train_images', '%s/%s'%(int(cls),id))       
def train_cls_path(root, id, cls):
    return os.path.join(root, 'test_images', '%s/%s'%(int(cls),id))  


def get_annotation(data_root):
    VOCdevkit_path  = os.path.join(data_root, "SCUT-FBP5500_v2")


    wd = getcwd()
    if not os.path.exists(os.path.join(VOCdevkit_path, "Classification")): os.makedirs(os.path.join(VOCdevkit_path, "Classification"))

    df = pd.read_excel(os.path.join(VOCdevkit_path, "All_Ratings.xlsx"), sheet_name="ALL") 
    df = df.groupby('Filename').median().reset_index()
    # df = df.round(0)
    df['Rating'] = df['Rating'].apply(lambda x : int(np.ceil(x)))

    df['image_path'] = df.apply(lambda x : train_img_path(VOCdevkit_path, x['Filename']),axis=1) 
    df['cls_path'] = df.apply(lambda x : train_cls_path(VOCdevkit_path, x['Filename'],x['Rating']),axis=1)         

    train_df, test_df = train_test_split(df, test_size=traintest)



    with tqdm(total=len(train_df)) as pbar:
        w = open(os.path.join(VOCdevkit_path, "Classification", "cls_train.txt"), "w")
        for i in range(len(train_df)):
            series = train_df.iloc[i]
            source, Rating = series['image_path'], str(series['Rating'])
            w.write(source + " " + Rating + "\n")
            pbar.update(1)
        w.close()
            
    with tqdm(total=len(test_df)) as pbar:
        w = open(os.path.join(VOCdevkit_path, "Classification", "cls_test.txt"), "w")
        for i in range(len(test_df)):
            series = test_df.iloc[i]
            source, Rating = series['image_path'], str(series['Rating'])
            w.write(source + " " + Rating + "\n")
            pbar.update(1) 
        w.close()
   
    
if __name__ == "__main__":
    data_root = '/home/leyan/DataSet/'
    get_annotation(data_root)   