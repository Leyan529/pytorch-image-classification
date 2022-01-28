import os
from os import getcwd
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
from glob import glob
import random
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

root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\HandWrite"

classes, _ = get_classes("model_data\\HandWrite_classes.txt")

traintest = 0.1

text_to_id = {}
id_to_text = {}

def img_path(id):
        img_path = os.path.join(root, '%s_images\\%s' % ('train', id))
        return img_path
def cls_path(id, cls):
    cls_path = os.path.join(root, '%s_images'%("test"), '%s\\%s'%(cls,id) )
    return cls_path 

if __name__ == "__main__":
    wd = getcwd()
    if not os.path.exists(os.path.join(root, "Classification")): os.makedirs(os.path.join(root, "Classification"))

    for idx, clas in enumerate(classes):
        clas = clas.strip()
        text_to_id[clas] = idx
        id_to_text[idx] = clas
        if not os.path.exists(os.path.join(root,"train", str(idx))):
            os.makedirs(os.path.join(root,"train",str(idx)))

    files = glob(os.path.join(root,"train","*.jpg"))
    random.shuffle(files)
    # indices = list(range(len(files)))
    # random.shuffle(indices)

    train_files = files[: int(len(files) * (1-traintest))]
    test_files = files[int(len(files) * (1-traintest)):]

    with tqdm(total=len(train_files)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_train.txt"), "w")
        for idx, source in enumerate(train_files):
            txt = source.split("_")[1].split(".")[0]
            Label = str(text_to_id[txt])
            fn = source.split("\\")[-1]
            
            # print(source)
            dest = os.path.join(root,"train", Label, fn)
            dest = dest.replace(txt, Label)
            shutil.copy(source,dest)
            w.write(dest + " " + Label + "\n")
            pbar.update(1)
        w.close()

    with tqdm(total=len(test_files)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_test.txt"), "w")
        for idx, source in enumerate(test_files):
            txt = source.split("_")[1].split(".")[0]
            Label = str(text_to_id[txt])
            fn = source.split("\\")[-1]
            
            # print(source)
            dest = os.path.join(root,"train", Label, fn)
            dest = dest.replace(txt, Label)
            shutil.copy(source,dest)
            w.write(dest + " " + Label + "\n")
            pbar.update(1)
        w.close() 