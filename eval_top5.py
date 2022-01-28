import numpy as np
import torch
from PIL import Image

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
import os

data_dir = "D:/WorkSpace/JupyterWorkSpace/DataSet/Image-Classification/EdgeAOI/"
# data_dir = "D:/WorkSpace/JupyterWorkSpace/DataSet/Image-Classification/HandWrite/"
# data_dir = "D:/WorkSpace/JupyterWorkSpace/DataSet/Image-Classification/SCUT-FBP5500_v2/"

class top5_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argsort(preds)[::-1]
        arg_pred_top5 = arg_pred[:5]
        return arg_pred_top5

def evaluteTop5(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path, y = line.strip().split(' ')       
        x = Image.open(annotation_path)
        y = int(y)
        pred = classfication.detect_image(x)
        correct += y in pred
        if index % 100 == 0:
            print("[%d/%d]"%(index,total))
    return correct / total

classfication = top5_Classification()
with open(os.path.join(data_dir, "Classification/cls_test.txt"),"r") as f:
    lines = f.readlines()
top5 = evaluteTop5(classfication, lines)
print("top-5 accuracy = %.2f%%" % (top5*100))

