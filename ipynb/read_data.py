import os
from PIL import Image
from torch.utils.data import DataSet
from torchvision import transforms

data_trans = transforms.ToTensor()
normal_trans = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

# Compose Resize
resize_trans = transforms.Resize((512,512))
comp_resize_trans = transforms.Compose([resize_trans, data_trans, normal_trans])

# Compose RandomCrop
rc_trans = transforms.RandomCrop(512)
comp_rc_trans = transforms.Compose([rc_trans, data_trans, normal_trans])


class MyData(DataSet):
    def __init__(self,root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self,idx):
        img_name - self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir     
        # Resize
        data = resize_trans(img)
        # ToTensor
        data = data_trans(data)
        # Normalize
        data = normal_trans(data)
        # Compose 
        data = comp_resize_trans(img)

        return data, label
    def __len__(self):
        return len(self.img_path)

class MyDataLabel(DataSet):
    def __init__(self,root_dir, category):
        self.root_dir = os.path.join(root_dir,category)
        self.img_path = os.listdir(self.path)
        self.label = category
    def __getitem__(self,idx):
        img_name - self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label     
        # Resize
        data = resize_trans(img)
        # ToTensor
        data = data_trans(data)
        # Normalize
        data = normal_trans(data)
        # Compose 
        data = comp_resize_trans(img)

        return data, label
    def __len__(self):
        return len(self.img_path)


'''root_dir = "dataset/train"
A_label_dir = "A"
B_label_dir = "B"
A_dataset = MyData(root_dir,A_label_dir)
B_dataset = MyData(root_dir,B_label_dir)
train_dataset = A_dataset + B_dataset'''

root_dir = "DataSet/MongoLeaf/train"
A_dataset = MyDataLabel(root_dir,'healthy')
B_dataset = MyDataLabel(root_dir,'diseased')
train_dataset = A_dataset + B_dataset