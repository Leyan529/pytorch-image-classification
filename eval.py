import os

import numpy as np
import torch
import argparse

from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import letterbox_image
from utils.utils_metrics import evaluteTop1_5
import importlib

class Eval_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image),
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
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

        return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')


    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)

    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)

    metrics_out_path        = os.path.join(opt.out_path, "metrics_out") 


    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)
            
    classfication = Eval_Classification(root_dir=opt.out_path, 
                                    classes_path=opt.classes_path, 
                                    input_shape=opt.input_shape,
                                    backbone=opt.net)
    
    # with open("./cls_test.txt","r") as f:
    #     lines = f.readlines()

    lines = opt.lines
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))