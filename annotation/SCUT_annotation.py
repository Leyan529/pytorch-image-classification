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

root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\SCUT-FBP5500_v2"

classes, _ = get_classes("model_data\\SCUT_classes.txt")

traintest = 0.1

text_to_id = {}
id_to_text = {}

def train_img_path(id):
    return os.path.join(root, 'Images\\%s' % (id)) 
def train_cls_path(id, cls):
    return os.path.join(root, 'train_images', '%s\\%s'%(int(cls),id))       
def train_cls_path(id, cls):
    return os.path.join(root, 'test_images', '%s\\%s'%(int(cls),id))  

if __name__ == "__main__":
    wd = getcwd()
    if not os.path.exists(os.path.join(root, "Classification")): os.makedirs(os.path.join(root, "Classification"))

    df = pd.read_excel(os.path.join(root, "All_Ratings.xlsx"), sheet_name="ALL") 
    df = df.groupby('Filename').median().reset_index()
    # df = df.round(0)
    df['Rating'] = df['Rating'].apply(lambda x : int(np.ceil(x)))

    df['image_path'] = df['Filename'].apply(train_img_path)
    df['cls_path'] = df.apply(lambda x : train_cls_path(x['Filename'],x['Rating']),axis=1)         

    train_df, test_df = train_test_split(df, test_size=traintest)



    with tqdm(total=len(train_df)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_train.txt"), "w")
        for i in range(len(train_df)):
            series = train_df.iloc[i]
            source, Rating = series['image_path'], str(series['Rating'])
            w.write(source + " " + Rating + "\n")
            pbar.update(1)
        w.close()
            
    with tqdm(total=len(test_df)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_test.txt"), "w")
        for i in range(len(test_df)):
            series = test_df.iloc[i]
            source, Rating = series['image_path'], str(series['Rating'])
            w.write(source + " " + Rating + "\n")
            pbar.update(1) 
        w.close()

    # df = shuffle(df)

    # train_df = df[:int(len(df) * 0.8)]
    # test_df = df[int(len(df) * 0.8):]

    # for idx, clas in enumerate(classes):
    #     clas = clas.strip()
    #     text_to_id[clas] = idx
    #     id_to_text[idx] = clas
    #     if not os.path.exists(os.path.join(root,"train", str(idx))):
    #         os.makedirs(os.path.join(root,"train",str(idx)))

    # files = glob(os.path.join(root,"train","*.jpg"))
    # random.shuffle(files)
    # # indices = list(range(len(files)))
    # # random.shuffle(indices)

    # train_files = files[: int(len(files) * (1-traintest))]
    # test_files = files[int(len(files) * (1-traintest)):]

    # with tqdm(total=len(train_files)) as pbar:
    #     w = open(os.path.join(root, "Classification", "cls_train.txt"), "w")
    #     for idx, source in enumerate(train_files):
    #         txt = source.split("_")[1].split(".")[0]
    #         Label = str(text_to_id[txt])
    #         fn = source.split("\\")[-1]
            
    #         # print(source)
    #         dest = os.path.join(root,"train", Label, fn)
    #         dest = dest.replace(txt, Label)
    #         shutil.copy(source,dest)
    #         w.write(dest + " " + Label + "\n")
    #         pbar.update(1)
    #     w.close()

    # with tqdm(total=len(test_files)) as pbar:
    #     w = open(os.path.join(root, "Classification", "cls_test.txt"), "w")
    #     for idx, source in enumerate(test_files):
    #         txt = source.split("_")[1].split(".")[0]
    #         Label = str(text_to_id[txt])
    #         fn = source.split("\\")[-1]
            
    #         # print(source)
    #         dest = os.path.join(root,"train", Label, fn)
    #         dest = dest.replace(txt, Label)
    #         shutil.copy(source,dest)
    #         w.write(dest + " " + Label + "\n")
    #         pbar.update(1)
    #     w.close()
    
  