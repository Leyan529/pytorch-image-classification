import os
from os import getcwd
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
#---------------------------------------------------#
#   训练自己的数据集的时候一定要注意修改classes
#   修改成自己数据集所区分的种类
#   
#   种类顺序需要和训练时用到的model_data下的txt一样
#---------------------------------------------------#
classes = ["normal", "void", "horizontal defect", "vertical defect", "edge defect", "particle"]
sets    = ["train", "test"]

root = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Image-Classification\\EdgeAOI"

traintest = 0.1

def cls_path(id, cls):
        cls_path = os.path.join(root, '%s_images'%("train"), '%s\\%s'%(cls,id) )
        return cls_path  

def img_path(id):
        img_path = os.path.join(root, '%s_images\\%s' % ('train', id))
        return img_path

if __name__ == "__main__":
    wd = getcwd()
    
    total_df = pd.read_csv(os.path.join(root, 'train.csv'))

    total_df['cls_path'] = total_df.apply(lambda x : cls_path(x['ID'],x['Label']),axis=1)
    total_df['path'] = total_df['ID'].apply(img_path)
    classes = list(total_df["Label"].unique())

    for imgSet in sets:
        for cls in classes:        
            if not os.path.exists(os.path.join(root, "Classification", imgSet, str(cls))):
                os.makedirs(os.path.join(root, "Classification", imgSet, str(cls)))

    train_df, test_df = train_test_split(total_df, test_size=traintest)


    
    with tqdm(total=len(train_df)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_train.txt"), "w")
        for i in range(len(train_df)):
            series = train_df.iloc[i]
            source, dest, Label = series['path'], series['cls_path'], str(series['Label'])
            dest = dest.replace("train_images", "Classification//train")
            shutil.copy(source, dest)
            w.write(dest + " " + Label + "\n")
            pbar.update(1)    
        w.close()

    with tqdm(total=len(test_df)) as pbar:
        w = open(os.path.join(root, "Classification", "cls_test.txt"), "w")
        for i in range(len(test_df)):
            series = test_df.iloc[i]
            source, dest, Label = series['path'], series['cls_path'], str(series['Label'])
            dest = dest.replace("train_images", "Classification//test")
            shutil.copy(source, dest)
            w.write(dest + " " + Label + "\n")
            pbar.update(1)  
        w.close()